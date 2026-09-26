"""The macOS wheel assembly: the tag, the RECORD, and the bundle's execute bit.

WHY THIS HAS ITS OWN TEST, and why it is a pure-Python one. The wheel this builds is
the SHIPPING artefact of the macOS presence tier — the thing an operator's machine
installs instead of the pure wheel — and every way it can go wrong is silent:

* a wrong tag means macOS hosts silently get the pure wheel with no helper at all;
* a helper installed WITHOUT its execute bit means ``lop operator init`` fails with
  ``PermissionError`` at the one moment the operator asked for a key;
* a ``RECORD`` that disagrees with the archive means an installer or auditor reading
  it sees a corrupt wheel;
* a bundle without the embedded profile means the kernel SIGKILLs the helper on first
  use, which is the failure mode with no error message at all.

None of those needs a Mac to detect, so CI on Linux runs this and the release job's
artefact check is the same property measured on the real wheel.
"""

from __future__ import annotations

import base64
import csv
import hashlib
import io
import sys
import zipfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "packaging" / "macos"))

import make_macos_wheel  # noqa: E402 — imported by path, it is a build tool

WHEEL_NAME = "toy_package-1.0.0-py3-none-any.whl"


def _toy_wheel(tmp_path: Path) -> Path:
    """A wheel shaped like the real pure one: package data plus a dist-info."""
    wheel = tmp_path / WHEEL_NAME
    dist_info = "toy_package-1.0.0.dist-info"
    files = {
        "toy_package/__init__.py": b"__version__ = '1.0.0'\n",
        "toy_package/data.txt": b"some package data\n",
        f"{dist_info}/WHEEL": (
            "Wheel-Version: 1.0\nGenerator: setuptools\nRoot-Is-Purelib: true\n"
            "Tag: py3-none-any\n"
        ).encode(),
        f"{dist_info}/METADATA": b"Metadata-Version: 2.1\nName: toy-package\nVersion: 1.0.0\n",
    }
    record = []
    with zipfile.ZipFile(wheel, "w") as zf:
        for name, data in files.items():
            zf.writestr(name, data)
            record.append(
                (
                    name,
                    "sha256="
                    + base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode(),
                    str(len(data)),
                )
            )
        rows = "\n".join(",".join(row) for row in record) + f"\n{dist_info}/RECORD,,\n"
        zf.writestr(f"{dist_info}/RECORD", rows)
    return wheel


def _fake_app(tmp_path: Path) -> Path:
    app = tmp_path / "lop-keyagent.app"
    macos = app / "Contents" / "MacOS"
    macos.mkdir(parents=True)
    (macos / "lop-keyagent").write_bytes(b"\xcf\xfa\xed\xfe a universal2 Mach-O, in spirit\n")
    (app / "Contents" / "Info.plist").write_bytes(b"<plist/>")
    (app / "Contents" / "embedded.provisionprofile").write_bytes(b"a profile, in spirit\n")
    return app


def test_the_macos_wheel_is_tagged_and_renamed_for_macos(tmp_path: Path) -> None:
    wheel = make_macos_wheel.retag(_toy_wheel(tmp_path), _fake_app(tmp_path), tmp_path)
    assert wheel.name == "toy_package-1.0.0-py3-none-macosx_11_0_universal2.whl"
    with zipfile.ZipFile(wheel) as zf:
        names = zf.namelist()
        metadata = zf.read("toy_package-1.0.0.dist-info/WHEEL").decode()
        # The dist-info directory is NOT renamed: measured on the real wheel, its name
        # carries no tag, and renaming it would hide the metadata from an installer.
        assert "toy_package-1.0.0.dist-info/METADATA" in names
        assert "toy_package-1.0.0.dist-info/RECORD" in names
    assert "Tag: py3-none-macosx_11_0_universal2" in metadata
    # NOT purelib any more: the wheel carries a Mach-O binary, and installing it into
    # purelib on a host whose platlib differs is how a wheel ends up in the wrong place.
    assert "Root-Is-Purelib: false" in metadata


def test_the_bundle_lands_where_the_client_looks_for_it_and_stays_executable(
    tmp_path: Path,
) -> None:
    """The path and the mode are the two properties the runtime depends on.

    ``keyagent.helper_bundle_path()`` resolves
    ``<package>/operator/macos/lop-keyagent.app``, and the client execs
    ``Contents/MacOS/lop-keyagent`` directly rather than through LaunchServices.
    """
    from local_operator.operator.macos import keyagent

    wheel = make_macos_wheel.retag(_toy_wheel(tmp_path), _fake_app(tmp_path), tmp_path)
    expected = make_macos_wheel.BUNDLE_DEST / "Contents" / "MacOS" / keyagent.EXECUTABLE_NAME
    with zipfile.ZipFile(wheel) as zf:
        infos = {info.filename: info for info in zf.infolist()}
        assert expected.as_posix() in infos
        assert (
            make_macos_wheel.BUNDLE_DEST / "Contents" / "embedded.provisionprofile"
        ).as_posix() in infos
        assert (make_macos_wheel.BUNDLE_DEST / "Contents" / "Info.plist").as_posix() in infos
        mode = (infos[expected.as_posix()].external_attr >> 16) & 0o777
        assert mode == 0o755, f"the helper would install mode {mode:o}"
        profile_mode = (
            infos[
                (make_macos_wheel.BUNDLE_DEST / "Contents" / "embedded.provisionprofile").as_posix()
            ].external_attr
            >> 16
        ) & 0o777
        assert profile_mode == 0o644
    # The runtime resolves it from the INSTALLED package, so the assertion that matters
    # is that the path this wheel writes is the path ``helper_bundle_path`` builds —
    # a mismatch would be a macOS install whose helper is present and never found.
    resolved = keyagent.helper_bundle_path()
    assert tuple(make_macos_wheel.BUNDLE_DEST.parts) == tuple(resolved.parts[-4:]), (
        f"the wheel writes {make_macos_wheel.BUNDLE_DEST} but the runtime looks for "
        f"{resolved.parts[-4:]}"
    )


def test_the_record_describes_what_is_actually_in_the_archive(tmp_path: Path) -> None:
    """A RECORD that disagrees with the archive is a wheel an auditor reads as corrupt.

    Every entry added here is hashed from the bytes that were written, and the dist-info
    directory rename has to reach the RECORD's own path too.
    """
    wheel = make_macos_wheel.retag(_toy_wheel(tmp_path), _fake_app(tmp_path), tmp_path)
    with zipfile.ZipFile(wheel) as zf:
        rows = list(csv.reader(io.StringIO(zf.read("toy_package-1.0.0.dist-info/RECORD").decode())))
        recorded = {row[0]: row for row in rows}
        for name in zf.namelist():
            assert name in recorded, f"{name} is in the archive but not in RECORD"
        for name, digest, size in ((row[0], row[1], row[2]) for row in rows):
            if not digest:
                continue  # RECORD's own row
            data = zf.read(name)
            expected = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
            assert digest == f"sha256={expected}", f"RECORD disagrees with {name}"
            assert size == str(len(data)), f"RECORD size disagrees with {name}"
        assert (
            "toy_package-1.0.0.dist-info/RECORD,,"
            in zf.read("toy_package-1.0.0.dist-info/RECORD").decode()
        )


def test_a_bundle_that_could_not_run_is_refused_rather_than_packaged(tmp_path: Path) -> None:
    """The two shapes that would install cleanly and then fail on the operator's Mac."""
    app = _fake_app(tmp_path)
    (app / "Contents" / "MacOS" / "lop-keyagent").unlink()
    with pytest.raises(SystemExit, match="no Contents/MacOS/lop-keyagent"):
        make_macos_wheel.retag(_toy_wheel(tmp_path), app, tmp_path)

    app2 = _fake_app(tmp_path / "other")
    (app2 / "Contents" / "embedded.provisionprofile").unlink()
    with pytest.raises(SystemExit, match="embedded.provisionprofile"):
        make_macos_wheel.retag(_toy_wheel(tmp_path), app2, tmp_path)


def test_the_pure_wheel_is_left_alone(tmp_path: Path) -> None:
    """Both dists are published, so the input must still be a valid wheel afterwards."""
    wheel = _toy_wheel(tmp_path)
    before = wheel.read_bytes()
    make_macos_wheel.retag(wheel, _fake_app(tmp_path), tmp_path)
    assert wheel.read_bytes() == before
    assert wheel.name == WHEEL_NAME
