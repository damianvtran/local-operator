#!/usr/bin/env python3
"""Retag the pure wheel into the macOS wheel, with the signed key agent inside it.

WHY A SCRIPT RATHER THAN `python -m wheel tags`. Two things have to happen at once and
only one of them is a tag rewrite:

1. **the bundle is injected** as package data at
   ``local_operator/operator/macos/lop-keyagent.app/``, which is where
   ``keyagent.helper_bundle_path()`` looks, **with the executable's mode preserved**.
   A wheel stores Unix permissions in each entry's external attributes, and a helper
   installed without its execute bit is a helper that cannot be run at all — the failure
   is ``PermissionError`` at the moment the operator asks for a signature, which is the
   worst possible moment to discover a packaging bug.
2. **the wheel is retagged** to ``py3-none-macosx_11_0_universal2`` and
   ``Root-Is-Purelib`` becomes ``false``, because the wheel now contains a universal
   Mach-O binary and therefore belongs in platlib. Measured with ``uv build --wheel`` on
   this macOS host: the output is ``py3-none-any`` regardless, because the package has no
   compiled Python extension — so the platform tag has to be set explicitly either way.

A THIRD THING THAT IS EASY TO FORGET. ``.dist-info/RECORD`` is rebuilt from the bytes
actually written, so every injected entry carries its real sha256 and size and RECORD's
own row stays empty. An installer that checks hashes (``pip install`` verifies nothing on
the way in, but ``wheel unpack``, ``pip download --no-deps -v`` and auditors do) would
otherwise see a wheel whose own manifest disagrees with its contents.

AND ONE THING *NOT* TO DO: the ``.dist-info`` DIRECTORY is not renamed, because its name
carries no tag — measured on the real wheel built here, it is
``local_operator-0.62.31.dist-info`` and the tag lives only in the FILENAME and in
WHEEL's ``Tag:`` line. Renaming it would turn a valid wheel into one whose metadata no
installer can find.

Usage::

    make_macos_wheel.py --wheel dist/local_operator-<v>-py3-none-any.whl \\
        --app /path/to/lop-keyagent.app --out-dir dist/

Prints the path of the wheel it wrote.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import sys
import zipfile
from pathlib import Path

#: Where the bundle lands inside the wheel, and therefore inside ``site-packages``.
BUNDLE_DEST = Path("local_operator") / "operator" / "macos" / "lop-keyagent.app"

#: The tag the macOS wheel carries. ``macosx_11_0`` matches Info.plist's
#: ``LSMinimumSystemVersion``; ``universal2`` is what ``-arch arm64 -arch x86_64``
#: produces, and both must agree with the binary or an Intel Mac installs a wheel it
#: cannot execute.
PLATFORM_TAG = "macosx_11_0_universal2"

#: Modes for injected entries: the executable must be executable, everything else a
#: plain readable file. 0644 for the profile, because it is read by the kernel.
_EXECUTABLE = 0o755
_REGULAR = 0o644


def _record_row(path: str, data: bytes) -> tuple[str, str, str]:
    digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
    return path, f"sha256={digest}", str(len(data))


def _app_entries(app: Path) -> dict[str, tuple[bytes, int]]:
    """The bundle's files, keyed by their destination inside the wheel.

    Refuses a bundle that is not shaped the way the client executes it: a wheel that
    ships an ``.app`` without ``Contents/MacOS/<executable>`` installs cleanly and then
    fails at the first verb, which is exactly the silent-broken-install class this whole
    change exists to remove.
    """
    executable = app / "Contents" / "MacOS" / "lop-keyagent"
    if not executable.is_file():
        raise SystemExit(f"{app} has no Contents/MacOS/lop-keyagent")
    if not (app / "Contents" / "embedded.provisionprofile").is_file():
        raise SystemExit(
            f"{app} carries no Contents/embedded.provisionprofile: without it the kernel "
            "SIGKILLs the helper on first use (measured), so this wheel must not ship"
        )
    entries: dict[str, tuple[bytes, int]] = {}
    for path in sorted(app.rglob("*")):
        if not path.is_file():
            continue
        mode = _EXECUTABLE if path == executable else _REGULAR
        entries[(BUNDLE_DEST / path.relative_to(app)).as_posix()] = (path.read_bytes(), mode)
    return entries


def _retagged(metadata: str) -> str:
    """WHEEL with the macOS tag and a non-purelib root, everything else untouched."""
    lines = []
    for line in metadata.splitlines():
        if line.startswith("Tag: "):
            lines.append(f"Tag: py3-none-{PLATFORM_TAG}")
        elif line.startswith("Root-Is-Purelib:"):
            lines.append("Root-Is-Purelib: false")
        else:
            lines.append(line)
    return "\n".join(lines) + "\n"


def retag(wheel: Path, app: Path, out_dir: Path) -> Path:
    """Write the macOS wheel into ``out_dir`` and return its path.

    The pure wheel is read and never modified: both dists are published for the same
    version, and an in-place edit would publish the macOS wheel twice and the pure one
    never.
    """
    suffix = "-py3-none-any.whl"
    if not wheel.name.endswith(suffix):
        raise SystemExit(f"{wheel.name} is not a py3-none-any wheel")
    destination = out_dir / f"{wheel.name[: -len(suffix)]}-py3-none-{PLATFORM_TAG}.whl"
    out_dir.mkdir(parents=True, exist_ok=True)
    injected = _app_entries(app)

    with zipfile.ZipFile(wheel) as src:
        names = src.namelist()
        wheel_metadata = next((n for n in names if n.endswith(".dist-info/WHEEL")), None)
        record_name = next((n for n in names if n.endswith(".dist-info/RECORD")), None)
        if wheel_metadata is None or record_name is None:
            raise SystemExit(f"{wheel.name} has no .dist-info/WHEEL or /RECORD")
        if any(n.startswith(f"{BUNDLE_DEST}/") for n in names):
            raise SystemExit(f"{wheel.name} already contains the key agent bundle")
        written: dict[str, tuple[bytes, int]] = {}
        for info in src.infolist():
            if info.filename == record_name:
                continue  # rebuilt last, from what is actually written
            data = src.read(info)
            if info.filename == wheel_metadata:
                data = _retagged(data.decode("utf-8")).encode("utf-8")
            written[info.filename] = (data, (info.external_attr >> 16) & 0o777 or _REGULAR)

    written.update(injected)
    record = (
        "\n".join(
            [",".join(_record_row(path, data)) for path, (data, _mode) in sorted(written.items())]
            + [f"{record_name},,"]
        )
        + "\n"
    )

    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as dst:
        for entry, (data, mode) in sorted(written.items()):
            info = zipfile.ZipInfo(entry, date_time=(1980, 1, 1, 0, 0, 0))
            info.external_attr = (mode & 0xFFFF) << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            dst.writestr(info, data)
        info = zipfile.ZipInfo(record_name, date_time=(1980, 1, 1, 0, 0, 0))
        info.external_attr = _REGULAR << 16
        dst.writestr(info, record)
    return destination


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Retag the pure wheel into the macOS wheel, with the signed key agent"
    )
    parser.add_argument("--wheel", required=True, type=Path, help="the pure py3-none-any wheel")
    parser.add_argument("--app", required=True, type=Path, help="the signed lop-keyagent.app")
    parser.add_argument(
        "--out-dir", required=True, type=Path, help="where to write the macOS wheel"
    )
    args = parser.parse_args(argv)
    destination = retag(args.wheel, args.app, args.out_dir)
    print(destination)
    return 0


if __name__ == "__main__":  # pragma: no cover — a build tool, exercised by its test
    sys.exit(main())
