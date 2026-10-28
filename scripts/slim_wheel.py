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

IDEMPOTENT: a second run finds only kept comments and rewrites identical bytes.

Usage::

    slim_wheel.py dist/local_operator-<v>-py3-none-any.whl

Prints the wheel's size before and after.
"""

from __future__ import annotations

import argparse
import ast
import base64
import hashlib
import io
import os
import re
import sys
import tempfile
import tokenize
import zipfile
from pathlib import Path

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
        slimmed = strip_comments(source)
        before = ast.dump(ast.parse(source, filename=name), include_attributes=True)
        after = ast.dump(ast.parse(slimmed, filename=name), include_attributes=True)
    except (SyntaxError, tokenize.TokenError, UnicodeError, ValueError) as exc:
        raise SlimError(f"{name}: cannot be tokenized/parsed: {exc}") from exc
    if before != after:
        raise SlimError(f"{name}: the stripped module's AST differs from the original")
    if source.count("\n") != slimmed.count("\n"):
        raise SlimError(f"{name}: the stripped module's line count differs")
    return slimmed.encode(encoding)


def _record_row(path: str, data: bytes) -> tuple[str, str, str]:
    # Same row shape as packaging/macos/make_macos_wheel.py: urlsafe base64 sha256
    # without padding, then the byte size.
    digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
    return path, f"sha256={digest}", str(len(data))


def slim_wheel(wheel: Path) -> tuple[int, int]:
    """Rewrite ``wheel`` in place; return its (before, after) size in bytes.

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

    fd, tmp_name = tempfile.mkstemp(dir=wheel.parent, prefix=".slim-", suffix=".whl")
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as dst:
            for info, data in written:
                out = zipfile.ZipInfo(info.filename, date_time=info.date_time)
                out.external_attr = info.external_attr
                out.compress_type = zipfile.ZIP_DEFLATED
                dst.writestr(out, data)
            # RECORD last, as wheel builders write it, with the source entry's metadata.
            source_record = next(i for i in infos if i.filename == record_name)
            out = zipfile.ZipInfo(record_name, date_time=source_record.date_time)
            out.external_attr = source_record.external_attr
            out.compress_type = zipfile.ZIP_DEFLATED
            dst.writestr(out, record)
        os.replace(tmp, wheel)
    finally:
        tmp.unlink(missing_ok=True)
    return before, wheel.stat().st_size


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Strip comments from a built py3-none-any wheel, in place, verified"
    )
    parser.add_argument("wheels", nargs="+", type=Path, help="py3-none-any wheel(s)")
    args = parser.parse_args(argv)
    for wheel in args.wheels:
        try:
            before, after = slim_wheel(wheel)
        except SlimError as exc:
            print(f"::error::slim_wheel: {exc}", file=sys.stderr)
            return 1
        saved = before - after
        print(f"{wheel.name}: {before:,} -> {after:,} bytes (-{saved:,}, {saved / before:.1%})")
    return 0


if __name__ == "__main__":  # pragma: no cover — a build tool, exercised by its test
    sys.exit(main())
