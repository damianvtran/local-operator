#!/usr/bin/env python3
"""Strip ``#`` comments from every module of a built pure wheel, in place, verified.

WHY THIS EXISTS. PyPI caps a project at 10 GB, and local-operator ships several
releases a day. Measured on the published 0.90.17, one release cost ~59 MB: the pure
wheel 19.98 MB, the ``macosx_11_0_universal2`` wheel 20.14 MB and the sdist 18.85 MB.
The cap was reached and v0.90.18/v0.90.19 failed with ``400 Project size too large``.
``.py`` files were 16.9 of the wheel's 19.7 MB, and this codebase's convention is long
WHY comments — valuable in the repository, dead weight in ``site-packages``. Stripping
them takes the wheel to ~14.7 MB; the sdist moved off PyPI to the GitHub Release (see
``publish.yml``), so a release drops from ~59 MB to ~29 MB.

WHAT IS KEPT, AND WHY:

* **every line, including its newline.** A comment-only line becomes an empty line,
  never a deleted one, so every traceback line number, ``inspect`` lookup and
  ``linecache`` read in an installed module still points at the same line as the
  repository's source. That is the property that keeps a user's traceback debuggable
  against the tagged commit.
* **code before an inline comment**, minus the whitespace that separated them.
* **docstrings**: they are not comments, they are runtime values (``__doc__``, ``help``,
  argparse/Typer help text), so they are never touched — ``tokenize`` sees them as
  ``STRING`` tokens.
* **the shebang** (line 1, ``#!``) and **PEP 263 coding lines** (lines 1-2): both are
  read by something other than the compiler's comment-skipping path.
* **pragma comments** — ``# type:``, ``# noqa``, ``# pragma``, ``# pyright:``,
  ``# fmt:`` — because a type checker or linter run against the INSTALLED package
  (a downstream project, an auditor) reads them, and they are a few bytes each.

THE SAFETY NET, and why it fails closed. Each rewritten module is checked against its
original: ``ast.dump(..., include_attributes=True)`` must be identical (so every node,
and every node's line AND column, is unchanged — the code is the same code at the same
positions) and the ``\\n`` count must be equal. A file that ``tokenize`` or
``ast.parse`` rejects, or that fails either comparison, aborts the whole run with a
non-zero exit naming the file and leaves the wheel untouched: a half-slimmed or
unverified wheel must never reach PyPI, where an upload is irreversible.

THE WHEEL STAYS A VALID WHEEL. Non-``.py`` members keep their bytes; every entry keeps
its ``date_time`` and ``external_attr`` (an executable bit lost here is a helper that
cannot run); ``.dist-info/RECORD`` is rebuilt from the bytes actually written, exactly
as ``packaging/macos/make_macos_wheel.py`` does, so its sha256/size rows agree with the
contents and RECORD's own row stays empty. The new wheel is written to a temporary file
in the same directory and moved over the original, so an interrupted run leaves the
original wheel, never a truncated one.

ZOPFLI DEFLATE, and why it is safe. Every member is stored as standard DEFLATE
(method 8, what every ``zipfile``/pip/installer already reads), but the DEFLATE stream
is produced by zopfli rather than zlib. zopfli searches far harder for the shortest
encoding of the SAME bytes; it is lossless by construction and changes only ZIP
framing, never a byte that lands in ``site-packages``. Measured on the 0.105.3 slim
wheel (CI run 43783144948): 18,348,918 -> 17,698,969 bytes (-649,949, 0.62 MiB) at one
iteration, about 185 s of CPU on one Apple-silicon core, byte-identical across two
independent runs. More iterations buy little: on a 1.5 MB sample, 15 iterations saved
only ~11% more than 1 for ~1.7x the CPU, so ``_ZOPFLI_ITERATIONS`` stays at 1. Members
are compressed in parallel worker processes (``--jobs``); each stream depends only on
its member's bytes, so the parallel result is byte-identical to the serial one.

zopfli is a hard dependency, pinned with hashes in ``scripts/slim_wheel-requirements.txt``
for the CI/release jobs and in the ``dev`` extra for the tests. There is deliberately no
silent zlib fallback: a release that quietly lost 0.6 MiB of headroom would first show
up as a red size budget, or not at all. After writing, the temporary wheel is re-read
and every member must decompress to exactly the bytes intended, with a matching CRC and
the zopfli stream's exact compressed size, or the run fails closed as above.

IDEMPOTENT: a second run finds only kept comments and rewrites identical bytes.

Usage::

    slim_wheel.py [--jobs N] dist/local_operator-<v>-py3-none-any.whl

Prints the wheel's size before and after.
"""

from __future__ import annotations

import argparse
import ast
import base64
import contextlib
import functools
import hashlib
import io
import os
import re
import sys
import tempfile
import tokenize
import zipfile
import zlib
from collections.abc import Iterator, Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import zopfli.zopfli

# Spend build CPU, not the fixed PyPI storage budget (see "ZOPFLI DEFLATE" above).
_ZOPFLI_ITERATIONS = 1

# zopfli's Python binding emits a zlib container: a 2-byte header and a 4-byte
# Adler-32 trailer around the raw DEFLATE stream a ZIP member carries.
_ZLIB_HEADER = 2
_ZLIB_TRAILER = 4

#: Comments a tool other than the compiler reads from installed code. Matched against
#: the comment token's text, so ``#noqa`` and ``#  type: ignore`` both count.
_PRAGMA = re.compile(r"#\s*(type:|noqa|pragma|pyright:|fmt:)")

#: PEP 263's own regular expression, applied only on lines 1-2 as the PEP specifies.
_CODING = re.compile(r"^[ \t\f]*#.*?coding[:=][ \t]*[-\w.]+")


class SlimError(Exception):
    """A module could not be slimmed AND verified; the wheel must not be rewritten."""


def _kept(text: str, row: int) -> bool:
    if row == 1 and text.startswith("#!"):
        return True
    if row <= 2 and _CODING.match(text):
        return True
    return bool(_PRAGMA.match(text))


def strip_comments(source: str) -> str:
    """``source`` with every non-kept comment removed and every line preserved.

    A comment always runs to the end of its physical line, so removing it is a cut at
    the comment's column; the whitespace before the cut goes too, and the line's own
    ending (``\\n`` or ``\\r\\n``) is re-attached unchanged.
    """
    # Split exactly as tokenize's readline does (on "\n" only), so a token's row indexes
    # the same line; str.splitlines would also break on \f, \x1c or \u2028 and shift
    # every later row.
    lines = io.StringIO(source).readlines()
    cuts: dict[int, int] = {}
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type == tokenize.COMMENT and not _kept(token.string, token.start[0]):
            cuts[token.start[0]] = token.start[1]
    for row, col in cuts.items():
        line = lines[row - 1]
        body = line.rstrip("\r\n")
        lines[row - 1] = body[:col].rstrip() + line[len(body) :]
    return "".join(lines)


def slim_module(name: str, data: bytes) -> bytes:
    """The slimmed bytes of one module, verified against the original or raising."""
    try:
        encoding, _ = tokenize.detect_encoding(io.BytesIO(data).readline)
        # `utf-8-sig` round-trips its BOM through decode/encode, so the file's first
        # bytes are preserved along with everything else.
        source = data.decode(encoding)
        out = strip_comments(source).encode(encoding)
        # BOTH SIDES ARE PARSED FROM BYTES, and the "after" side from the exact bytes
        # that get written: the compiler then does its own PEP 263/BOM detection, so a
        # lost or altered coding cookie, or an encode round trip that changed a byte,
        # fails here instead of at import time on a user's machine. Parsing the decoded
        # text would skip that path entirely (str input ignores the cookie).
        before = ast.dump(ast.parse(data, filename=name), include_attributes=True)
        after = ast.dump(ast.parse(out, filename=name), include_attributes=True)
    except (SyntaxError, tokenize.TokenError, UnicodeError, ValueError) as exc:
        raise SlimError(f"{name}: cannot be tokenized/parsed: {exc}") from exc
    if before != after:
        raise SlimError(f"{name}: the stripped module's AST differs from the original")
    # Byte-level count is exact: PEP 263 encodings are ASCII-compatible, so b"\n" is
    # the newline in every encoding tokenize accepts.
    if data.count(b"\n") != out.count(b"\n"):
        raise SlimError(f"{name}: the stripped module's line count differs")
    return out


def _zopfli_streams(payloads: Sequence[bytes], jobs: int) -> list[bytes]:
    """The raw DEFLATE stream of each payload, in order.

    The worker is ``functools.partial`` over zopfli's C function, not a function of this
    module: the tests load this file with ``runpy``, whose functions cannot be pickled
    into a worker process, while a builtin pickles by its import path.
    """
    compress = functools.partial(zopfli.zopfli.compress, numiterations=_ZOPFLI_ITERATIONS)
    if jobs <= 1 or len(payloads) <= 1:
        wrapped = [compress(data) for data in payloads]
    else:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            # Chunked so ~3,000 small members do not pay one IPC round trip each.
            wrapped = list(pool.map(compress, payloads, chunksize=16))
    return [z[_ZLIB_HEADER:-_ZLIB_TRAILER] for z in wrapped]


class _PrecompressedDeflate:
    """Stands in for ``zlib.compressobj`` inside ``zipfile``: emits a ready stream.

    ``zipfile`` still computes the CRC and the uncompressed size from the bytes it is
    given, so only the compressed bytes come from here; ``flush`` refuses a payload
    whose length differs from the one the stream was made for, so a stream can never be
    attached to the wrong member silently.
    """

    def __init__(self, stream: bytes, size: int) -> None:
        self._stream, self._size, self._seen = stream, size, 0

    def compress(self, data: bytes) -> bytes:
        self._seen += len(data)
        return b""

    def flush(self) -> bytes:
        if self._seen != self._size:
            raise SlimError(f"zopfli stream for {self._size} bytes given {self._seen}")
        return self._stream


@contextlib.contextmanager
def _zopfli_deflate(streams: list[tuple[bytes, int]]) -> Iterator[None]:
    """Make each DEFLATE member written in this block use the next stream, in order.

    ``zipfile`` has no public way to write a precompressed member, so this swaps its
    module-level ``_get_compressor`` (the same seam the tests already observe) for the
    duration of the write only, and restores it even on failure. Every stream must be
    consumed exactly once; anything else means the hook was bypassed (a changed
    ``zipfile``) and the wheel would silently carry zlib output.
    """
    original = getattr(zipfile, "_get_compressor")
    pending = iter(streams)

    def precompressed(compress_type: int, compresslevel: int | None = None) -> object:
        if compress_type != zipfile.ZIP_DEFLATED:
            return original(compress_type, compresslevel)
        try:
            stream, size = next(pending)
        except StopIteration:
            raise SlimError("more DEFLATE members written than zopfli streams") from None
        return _PrecompressedDeflate(stream, size)

    setattr(zipfile, "_get_compressor", precompressed)
    try:
        yield
    finally:
        setattr(zipfile, "_get_compressor", original)
    if next(pending, None) is not None:
        raise SlimError("zopfli streams left unused: zipfile bypassed the compressor hook")


def _verify_written(path: Path, expected: list[tuple[str, bytes, int]]) -> None:
    """Fail closed unless ``path`` holds exactly ``expected`` (name, bytes, deflated size)."""
    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
        if [i.filename for i in infos] != [name for name, _, _ in expected]:
            raise SlimError("the written wheel's member list differs from the intended one")
        for info, (name, data, deflated) in zip(infos, expected):
            if info.compress_type != zipfile.ZIP_DEFLATED or info.compress_size != deflated:
                raise SlimError(f"{name}: not stored as the zopfli DEFLATE stream")
            try:
                # read() also checks the stream against the stored CRC.
                intact = info.CRC == zlib.crc32(data) and archive.read(info) == data
            except (zipfile.BadZipFile, zlib.error, EOFError) as exc:
                raise SlimError(f"{name}: does not decompress: {exc}") from exc
            if not intact:
                raise SlimError(f"{name}: does not decompress to the intended bytes")


def _record_row(path: str, data: bytes) -> tuple[str, str, str]:
    # Same row shape as packaging/macos/make_macos_wheel.py: urlsafe base64 sha256
    # without padding, then the byte size.
    digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
    return path, f"sha256={digest}", str(len(data))


def slim_wheel(wheel: Path, jobs: int = 1) -> tuple[int, int]:
    """Rewrite ``wheel`` in place; return its (before, after) size in bytes.

    ``jobs`` worker processes run zopfli; the output is identical for any value.

    Every module is slimmed and verified BEFORE anything is written, so a failure
    raises :class:`SlimError` with the original wheel untouched.
    """
    if not wheel.name.endswith("-py3-none-any.whl"):
        raise SlimError(f"{wheel.name} is not a py3-none-any wheel")
    before = wheel.stat().st_size
    with zipfile.ZipFile(wheel) as src:
        infos = src.infolist()
        record_name = next(
            (i.filename for i in infos if i.filename.endswith(".dist-info/RECORD")), None
        )
        if record_name is None:
            raise SlimError(f"{wheel.name} has no .dist-info/RECORD")
        written: list[tuple[zipfile.ZipInfo, bytes]] = []
        for info in infos:
            if info.filename == record_name or info.is_dir():
                continue
            data = src.read(info)
            if info.filename.endswith(".py"):
                data = slim_module(info.filename, data)
            written.append((info, data))

    record = "".join(",".join(_record_row(info.filename, data)) + "\n" for info, data in written)
    record += f"{record_name},,\n"
    source_record = next(i for i in infos if i.filename == record_name)
    members = list(written)
    members.append((source_record, record.encode()))
    streams = _zopfli_streams([data for _, data in members], jobs)

    fd, tmp_name = tempfile.mkstemp(dir=wheel.parent, prefix=".slim-", suffix=".whl")
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        sized = [(stream, len(data)) for stream, (_, data) in zip(streams, members)]
        with _zopfli_deflate(sized), zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as dst:
            # RECORD is last, as wheel builders write it, with the source entry's metadata.
            for info, data in members:
                out = zipfile.ZipInfo(info.filename, date_time=info.date_time)
                out.external_attr = info.external_attr
                out.compress_type = zipfile.ZIP_DEFLATED
                dst.writestr(out, data)
        _verify_written(tmp, [(i.filename, d, len(s)) for (i, d), s in zip(members, streams)])
        # mkstemp creates 0600 and os.replace keeps the temp file's mode, so without
        # this a slimmed wheel would be owner-only (measured: -rw------- after vs
        # -rw-r--r-- before). Keep the source wheel's permission bits.
        os.chmod(tmp, wheel.stat().st_mode & 0o777)
        os.replace(tmp, wheel)
    finally:
        tmp.unlink(missing_ok=True)
    return before, wheel.stat().st_size


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Strip comments from a built py3-none-any wheel, in place, verified"
    )
    parser.add_argument("wheels", nargs="+", type=Path, help="py3-none-any wheel(s)")
    parser.add_argument(
        "--jobs",
        type=int,
        default=os.cpu_count() or 1,
        help="zopfli worker processes (default: CPU count; output is identical for any N)",
    )
    args = parser.parse_args(argv)
    for wheel in args.wheels:
        try:
            before, after = slim_wheel(wheel, jobs=args.jobs)
        except SlimError as exc:
            print(f"::error::slim_wheel: {exc}", file=sys.stderr)
            return 1
        saved = before - after
        print(f"{wheel.name}: {before:,} -> {after:,} bytes (-{saved:,}, {saved / before:.1%})")
    return 0


if __name__ == "__main__":  # pragma: no cover — a build tool, exercised by its test
    sys.exit(main())
