#!/usr/bin/env python3
"""Checkout-only ordered audit storage; never part of the runtime catalogue API.

The immutable marker pins legacy BYTES. Per-key suffixes avoid a mutable shared
index; their effective-history digest also forces divergent same-key Git edits
to conflict. Digests establish integrity, never recency or shipping eligibility.
All APIs take an explicit checkout root, including isolated fixture roots.
"""

from __future__ import annotations

import argparse
import errno
import hashlib
import json
import os
import re
import stat
import subprocess
import sys
import tempfile
from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

MARKER = "i18n/audit-history.json"
AGGREGATE = "i18n/ledger.json"
SHARDS = "i18n/ledger"
HISTORY = "i18n/audit-history"
COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
SHA256 = re.compile(r"^[a-f0-9]{64}$")
Row = dict[str, Any]
Key = tuple[str, str]


def _canonical(value: Any) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )


def _same_history(left: list[Row], right: list[Row]) -> bool:
    # Python equality collapses true/1/1.0. Audit identity must match the exact
    # canonical JSON domain used by the integrity digest, including extensions.
    return _canonical(left) == _canonical(right)


def _same_groups(left: dict[Key, list[Row]], right: dict[Key, list[Row]]) -> bool:
    return left.keys() == right.keys() and all(
        _same_history(history, right[key]) for key, history in left.items()
    )


def history_sha256(rows: list[Row]) -> str:
    return hashlib.sha256(_canonical(rows).encode("utf-8")).hexdigest()


def _members(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _safe(path: Path, root: Path) -> None:
    """Reject symlinks at every component, including dangling links."""
    relative = path.absolute().relative_to(root.absolute())
    current = root.absolute()
    for part in (".", *relative.parts):
        current = current / part
        if current.is_symlink():
            raise ValueError(f"symlink audit path refused: {path}")


def _read(path: Path, root: Path) -> Any:
    _safe(path, root)
    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_members,
            parse_constant=lambda value: _invalid_constant(value),
        )
    except (OSError, UnicodeError, ValueError) as exc:
        raise ValueError(f"invalid audit JSON {path}: {exc}") from exc


def _invalid_constant(value: str) -> Any:
    raise ValueError(f"non-JSON number: {value}")


def _component(value: Any) -> str:
    if not isinstance(value, str) or not COMPONENT.fullmatch(value):
        raise ValueError(f"invalid audit key component: {value!r}")
    return value


def _rows(payload: Any, *, locale: str | None = None, namespace: str | None = None) -> list[Row]:
    if (
        not isinstance(payload, dict)
        or type(payload.get("schema")) is not int
        or payload["schema"] != 1
    ):
        raise ValueError("unsupported audit schema (expected 1)")
    rows = payload.get("entries")
    if not isinstance(rows, list):
        raise ValueError("audit entries must be a list")
    if locale is not None and "locale" in payload and payload["locale"] != locale:
        raise ValueError("wrong embedded locale")
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("audit receipt must be an object")
        key = (_component(row.get("locale")), _component(row.get("namespace")))
        if (locale is not None and key[0] != locale) or (
            namespace is not None and key[1] != namespace
        ):
            raise ValueError("wrong embedded audit key")
        if not isinstance(row.get("source_sha256"), str) or not SHA256.fullmatch(
            row["source_sha256"]
        ):
            raise ValueError("invalid audit source_sha256")
        if not isinstance(row.get("status"), str) or not row["status"]:
            raise ValueError("invalid audit status")
        # Some immutable legacy translator identities were explicitly null.
        # Preserve that admitted JSON shape rather than fabricate provenance;
        # identity truth/completeness remains the independent audit gate.
        for role in ("translator", "auditor"):
            provenance = row.get(role)
            if not isinstance(provenance, dict) or any(
                field not in provenance
                or (provenance[field] is not None and not isinstance(provenance[field], str))
                for field in ("agent", "model")
            ):
                raise ValueError(f"invalid audit {role} provenance")
    _unique(rows)
    return rows


def _unique(rows: list[Row]) -> None:
    seen: set[str] = set()
    for row in rows:
        identity = _canonical(row)
        if identity in seen:
            raise ValueError(f"duplicate full audit receipt: {row['locale']}/{row['namespace']}")
        seen.add(identity)


def _group(rows: list[Row]) -> dict[Key, list[Row]]:
    grouped: dict[Key, list[Row]] = defaultdict(list)
    for row in rows:
        grouped[(row["locale"], row["namespace"])].append(row)
    return dict(grouped)


def _shard_paths(root: Path) -> list[Path]:
    directory = root / SHARDS
    _safe(directory, root)
    if not directory.exists():
        return []
    paths = sorted(directory.iterdir())
    for path in paths:
        _safe(path, root)
        if not path.is_file() or path.suffix != ".json":
            raise ValueError(f"unexpected legacy shard path: {path}")
        _component(path.stem)
    return paths


def _legacy(
    root: Path, *, allow_stranded: bool = False, require_agreement: bool = False
) -> tuple[list[Row], dict[str, str]]:
    paths = _shard_paths(root)
    aggregate = root / AGGREGATE
    inventory: dict[str, str] = {}
    if aggregate.exists() or aggregate.is_symlink():
        aggregate_rows = _rows(_read(aggregate, root))
        inventory[AGGREGATE] = hashlib.sha256(aggregate.read_bytes()).hexdigest()
    else:
        aggregate_rows = []
    rows: list[Row] = []
    for path in paths:
        rows.extend(_rows(_read(path, root), locale=path.stem))
        inventory[path.relative_to(root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    if not paths:
        rows = aggregate_rows
    elif aggregate.exists():
        # Aggregate mirrors may differ in cross-key ordering, but neither extra
        # receipts nor per-key reordering can be silently selected away.
        aggregate_groups, shard_groups = _group(aggregate_rows), _group(rows)
        agrees = (
            _same_groups(aggregate_groups, shard_groups)
            if require_agreement
            else all(
                _same_history(shard_groups.get(key, [])[: len(history)], history)
                for key, history in aggregate_groups.items()
            )
        )
        if not agrees and not allow_stranded:
            raise ValueError(
                "stranded or reordered aggregate receipts; reconcile with "
                "shard_ledger.py --fold-stranded before migration"
            )
    _unique(rows)
    return rows, inventory


def _suffix_paths(root: Path) -> list[Path]:
    directory = root / HISTORY
    _safe(directory, root)
    if not directory.exists():
        return []
    result: list[Path] = []
    for locale in sorted(directory.iterdir()):
        _safe(locale, root)
        _component(locale.name)
        if not locale.is_dir():
            raise ValueError(f"unexpected audit locale path: {locale}")
        for path in sorted(locale.iterdir()):
            _safe(path, root)
            if not path.is_file() or path.suffix != ".json":
                raise ValueError(f"unexpected audit history path: {path}")
            _component(path.stem)
            result.append(path)
    return result


def migrated(root: Path) -> bool:
    path = root / MARKER
    _safe(path, root)
    return path.exists()


def load_entries(root: Path, *, allow_stranded: bool = False) -> list[Row]:
    root = Path(root)
    suffixes = _suffix_paths(root)
    if not migrated(root):
        if suffixes:
            raise ValueError("audit history exists without its migration marker")
        return _legacy(root, allow_stranded=allow_stranded)[0]
    marker = _read(root / MARKER, root)
    if (
        not isinstance(marker, dict)
        or type(marker.get("schema")) is not int
        or marker.get("schema") != 1
        or marker.get("storage") != "ordered-suffix-v1"
    ):
        raise ValueError("unsupported audit migration marker schema")
    expected = marker.get("baseline")
    if (
        not isinstance(expected, dict)
        or AGGREGATE not in expected
        or any(
            not isinstance(value, str) or not SHA256.fullmatch(value) for value in expected.values()
        )
    ):
        raise ValueError("invalid audit baseline inventory")
    actual_paths = {AGGREGATE, *(p.relative_to(root).as_posix() for p in _shard_paths(root))}
    if set(expected) != actual_paths:
        raise ValueError(
            "audit baseline inventory mismatch; use import-legacy with a detached source"
        )
    for relative, digest in expected.items():
        path = root / relative
        _safe(path, root)
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(
                f"stale audit baseline {relative}; use import-legacy with a detached source"
            )
    rows, _ = _legacy(root, require_agreement=True)
    grouped = _group(rows)
    for path in suffixes:
        key = (path.parent.name, path.stem)
        payload = _read(path, root)
        added = _rows(payload, locale=key[0], namespace=key[1])
        if payload.get("locale") != key[0] or payload.get("namespace") != key[1] or not added:
            raise ValueError(f"wrong embedded key or empty suffix: {path}")
        effective = grouped.get(key, []) + added
        _unique(effective)
        if payload.get("history_sha256") != history_sha256(effective):
            raise ValueError(f"audit history digest mismatch: {path}")
        grouped[key] = effective
        rows.extend(added)
    _unique(rows)
    return rows


def export_aggregate(root: Path, *, allow_stranded: bool = False) -> str:
    rows = load_entries(root, allow_stranded=allow_stranded)
    return json.dumps({"schema": 1, "entries": rows}, indent=2, ensure_ascii=False) + "\n"


def _atomic(path: Path, payload: Any, root: Path) -> None:
    _safe(path, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _mutation_lock_path(root: Path) -> Path:
    # Coordination is checkout-local, never an audit receipt/shared index edit.
    # Git's per-worktree metadata keeps the durable lock out of tracked files;
    # detached non-Git fixture/export roots use an untracked local sidecar.
    # Read the checkout's explicit metadata pointer, not Git's ambient
    # GIT_DIR/HOME/PATH discovery: those must never split one store's lock.
    metadata = root / ".git"
    _safe(metadata, root)
    if metadata.is_dir():
        return metadata / "audit-history-write.lock"
    if metadata.is_file():
        pointer = metadata.read_text(encoding="utf-8").strip()
        if not pointer.startswith("gitdir: "):
            raise ValueError("invalid checkout metadata for audit mutation lock")
        directory = Path(pointer[len("gitdir: ") :])
        if not directory.is_absolute():
            directory = root / directory
        if not directory.is_dir():
            raise ValueError("missing checkout metadata for audit mutation lock")
        return directory / "audit-history-write.lock"
    return root / ".audit-history-write.lock"


@contextmanager
def _mutation(root: Path, *, apply: bool) -> Iterator[None]:
    """Serialize fresh read/check/replace across ALL cooperating local writers.

    Match config_writer's nonblocking flock/msvcrt primitive without importing
    runtime configuration into this standalone checkout tool. Atomic rename is
    not compare-and-swap. Keep the sidecar inode forever: unlink-on-release lets
    contenders lock different inodes and both acknowledge a lost update.
    """
    if not apply:
        yield
        return
    _safe(root / "i18n", root)
    path = _mutation_lock_path(root)
    _safe(path, path.parent)
    flags = (
        os.O_CREAT
        | os.O_RDWR
        | getattr(os, "O_BINARY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    try:
        fd = os.open(path, flags, 0o600)
    except OSError as exc:
        raise ValueError(f"unsafe audit mutation lock {path}: {exc}") from exc
    acquired = False
    try:
        opened = os.fstat(fd)
        named = path.stat(follow_symlinks=False)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or (opened.st_dev, opened.st_ino) != (named.st_dev, named.st_ino)
        ):
            raise ValueError(f"unsafe audit mutation lock target: {path}")
        try:
            if os.name == "nt":
                import msvcrt

                if opened.st_size == 0:
                    os.write(fd, b"\0")
                os.lseek(fd, 0, os.SEEK_SET)
                msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            if exc.errno in (errno.EACCES, errno.EAGAIN, errno.EWOULDBLOCK, errno.EDEADLK):
                raise ValueError("audit storage mutation is busy; retry shortly") from exc
            raise
        acquired = True
        yield
    finally:
        try:
            if acquired:
                if os.name == "nt":
                    import msvcrt

                    os.lseek(fd, 0, os.SEEK_SET)
                    msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


def migrate(root: Path, *, apply: bool = False) -> dict[str, Any]:
    with _mutation(root, apply=apply):
        rows = load_entries(root)
        if migrated(root):
            return {"state": "already migrated", "receipts": len(rows), "keys": len(_group(rows))}
        _, inventory = _legacy(root, require_agreement=True)
        if AGGREGATE not in inventory:
            raise ValueError("migration requires the legacy aggregate snapshot")
        marker = {"schema": 1, "storage": "ordered-suffix-v1", "baseline": inventory}
        report = {
            "state": "apply" if apply else "dry-run",
            "receipts": len(rows),
            "keys": len(_group(rows)),
            "marker": marker,
        }
        if apply:
            # The only migration write is the marker, atomically published LAST.
            _atomic(root / MARKER, marker, root)
        return report


def _write_suffix(root: Path, key: Key, effective: list[Row], baseline: list[Row]) -> str:
    path = root / HISTORY / key[0] / f"{key[1]}.json"
    _atomic(
        path,
        {
            "schema": 1,
            "locale": key[0],
            "namespace": key[1],
            "history_sha256": history_sha256(effective),
            "entries": effective[len(baseline) :],
        },
        root,
    )
    return path.relative_to(root).as_posix()


def append(root: Path, row: Row, *, expected_history_sha: str, apply: bool = False) -> str:
    with _mutation(root, apply=apply):
        current = _group(load_entries(root))
        if not migrated(root):
            raise ValueError("append requires migration marker; run migrate --apply")
        _rows({"schema": 1, "entries": [row]})
        key = (row["locale"], row["namespace"])
        history = current.get(key, [])
        if history_sha256(history) != expected_history_sha:
            raise ValueError(f"stale expected history for {key[0]}/{key[1]}")
        _unique(history + [row])
        baseline = _group(_legacy(root)[0]).get(key, [])
        if apply:
            return _write_suffix(root, key, history + [row], baseline)
        return f"{HISTORY}/{key[0]}/{key[1]}.json"


def import_legacy(root: Path, source: Path, format: str, *, apply: bool = False) -> list[str]:
    with _mutation(root, apply=apply):
        current = _group(load_entries(root))
        if not migrated(root):
            raise ValueError("import-legacy requires migration marker")
        source = source.absolute()
        # Never treat a changed destination snapshot as detached source authority.
        protected = [root / AGGREGATE, root / MARKER, root / SHARDS, root / HISTORY]
        if any(
            source.resolve() == p.resolve() or p.resolve() in source.resolve().parents
            for p in protected
        ):
            raise ValueError("import-legacy requires an explicit detached source")
        source_root = (
            source
            if source.is_dir()
            else (
                source.parent.parent
                if source.name == "ledger.json" and source.parent.name == "i18n"
                else None
            )
        )
        if source_root is not None:
            if migrated(source_root) or _suffix_paths(source_root):
                raise ValueError(
                    "import source has mixed/new authority; export current history explicitly"
                )
            if (source_root / AGGREGATE).exists() and _shard_paths(source_root):
                _legacy(source_root, require_agreement=True)
        if format == "aggregate":
            source_file = source / AGGREGATE if source.is_dir() else source
            incoming = _rows(_read(source_file, source_file.parent))
        else:
            incoming = []
            for path in _shard_paths(source):
                incoming.extend(_rows(_read(path, source), locale=path.stem))
            if not incoming:
                raise ValueError("source has no locale-shard receipts")
            # Explicit format chooses authority, but disagreement still requires
            # reconciliation: never turn that selection into a row-loss escape.
            if (source / AGGREGATE).exists():
                _legacy(source)
        planned: list[tuple[Key, list[Row]]] = []
        for key, history in sorted(_group(incoming).items()):
            existing = current.get(key, [])
            shared = min(len(existing), len(history))
            if not _same_history(existing[:shared], history[:shared]):
                raise ValueError(
                    f"non-prefix competing history: {key[0]}/{key[1]} "
                    f"(current {history_sha256(existing)}, source {history_sha256(history)})"
                )
            if len(history) > len(existing):
                planned.append((key, history))
        baseline = _group(_legacy(root)[0])
        result = []
        for key, history in planned:
            if apply:
                result.append(_write_suffix(root, key, history, baseline.get(key, [])))
            else:
                result.append(
                    # i18n: ignore machine path; i18n tooling owns this while checkout-only
                    f"{HISTORY}/{key[0]}/{key[1]}.json",
                )
        return result


def validate_base(root: Path, ref: str) -> None:
    """Compare to committed authority, not a recomputable head-only checksum.

    Materialize ONLY the small audit storage tree, never executable archives or
    the checkout. This keeps legacy and migrated readers on the same strict path.
    """

    def git(*args: str) -> bytes:
        result = subprocess.run(["git", "-C", str(root), *args], capture_output=True)
        if result.returncode:
            raise ValueError(
                f"audit base {ref!r} is unreadable: "
                f"{result.stderr.decode(errors='replace').strip()}"
            )
        return result.stdout

    git("rev-parse", "--verify", f"{ref}^{{commit}}")
    listing = git("ls-tree", "-rz", ref, "--", AGGREGATE, SHARDS, MARKER, HISTORY)
    with tempfile.TemporaryDirectory(prefix="audit-base-") as temporary:
        base = Path(temporary)
        for record in listing.split(b"\0"):
            if not record:
                continue
            metadata, name = record.split(b"\t", 1)
            mode, kind, oid = metadata.split()
            relative = name.decode("utf-8")
            if mode != b"100644" or kind != b"blob":
                raise ValueError(f"unsupported audit base file mode: {relative}")
            path = base / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(git("cat-file", "blob", oid.decode()))
        old = _group(load_entries(base))
        new = _group(load_entries(root))
        if migrated(base):
            if not migrated(root):
                raise ValueError("audit migration marker removal is forbidden")
            paths = [MARKER, *_read(base / MARKER, base)["baseline"]]
            for relative in paths:
                if (base / relative).read_bytes() != (root / relative).read_bytes():
                    raise ValueError(f"immutable audit baseline changed: {relative}")
            old_paths = {p.relative_to(base) for p in _suffix_paths(base)}
            new_paths = {p.relative_to(root) for p in _suffix_paths(root)}
            if not old_paths <= new_paths:
                raise ValueError("audit suffix file deletion is forbidden")
        elif migrated(root):
            # A one-time migration may pin a newer actual base only after full
            # lossless reconstruction; it cannot bless edits hidden in snapshots.
            if not _same_groups(_group(_legacy(root)[0]), old):
                raise ValueError("first migration baseline differs from actual audit base history")
            if _legacy(base)[1] != _legacy(root)[1]:
                raise ValueError("first migration must preserve actual-base snapshot bytes")
        for key, history in old.items():
            if not _same_history(new.get(key, [])[: len(history)], history):
                raise ValueError(f"audit history is not append-only: {key[0]}/{key[1]}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    sub = parser.add_subparsers(dest="command", required=True)
    history = sub.add_parser(
        "history-sha",
        # i18n: ignore developer-only CLI help; i18n tooling owns this while checkout-only
        help="expected effective-history digest for an append",
    )
    history.add_argument("--locale", required=True)
    history.add_argument("--namespace", required=True)
    export = sub.add_parser("export")
    export.add_argument("--output", type=Path)
    for command in ("migrate", "import-legacy", "append"):
        child = sub.add_parser(command)
        mode = child.add_mutually_exclusive_group(required=True)
        mode.add_argument("--dry-run", action="store_true")
        mode.add_argument("--apply", action="store_true")
        if command == "import-legacy":
            child.add_argument("--source", type=Path, required=True)
            child.add_argument("--format", choices=("aggregate", "locale-shards"), required=True)
        if command == "append":
            child.add_argument("--receipt", type=Path, required=True)
            child.add_argument("--expected-history-sha", required=True)
    validate = sub.add_parser("validate")
    validate.add_argument("--base", required=True)
    args = parser.parse_args(argv)
    root = args.root.absolute()
    try:
        if args.command == "export":
            content = export_aggregate(root)
            if args.output is None:
                sys.stdout.write(content)
            else:
                output = args.output.absolute()
                # resolve() catches symlink aliases, but not hard links. Refuse
                # those too: exporting through another name must never mutate a
                # pinned snapshot or an authoritative suffix inode.
                source_paths = [
                    root / AGGREGATE,
                    root / MARKER,
                    *_shard_paths(root),
                    *_suffix_paths(root),
                ]
                if output.exists() and any(
                    path.exists() and output.samefile(path) for path in source_paths
                ):
                    raise ValueError("export cannot overwrite audit source or immutable snapshots")
                for protected in (root / AGGREGATE, root / SHARDS, root / MARKER, root / HISTORY):
                    if (
                        output.resolve() == protected.resolve()
                        or protected.resolve() in output.resolve().parents
                    ):
                        raise ValueError(
                            "export cannot overwrite audit source or immutable snapshots"
                        )
                output.write_text(content, encoding="utf-8")
        elif args.command == "history-sha":
            key = (_component(args.locale), _component(args.namespace))
            print(history_sha256(_group(load_entries(root)).get(key, [])))
        elif args.command == "migrate":
            print(json.dumps(migrate(root, apply=args.apply), indent=2))
        elif args.command == "import-legacy":
            print(
                json.dumps(
                    {"paths": import_legacy(root, args.source, args.format, apply=args.apply)}
                )
            )
        elif args.command == "append":
            row = _read(args.receipt.absolute(), args.receipt.absolute().parent)
            print(
                append(root, row, expected_history_sha=args.expected_history_sha, apply=args.apply)
            )
        else:
            validate_base(root, args.base)
            print(
                # i18n: ignore developer-only diagnostic; i18n tooling owns this while checkout-only
                "audit history: valid and append-only",
            )
    except (ValueError, OSError) as exc:
        print(
            # i18n: ignore developer-only diagnostic; i18n tooling owns this while checkout-only
            f"ledger.py: {exc}",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
