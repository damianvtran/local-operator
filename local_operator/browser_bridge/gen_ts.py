"""Generate the TypeScript halves of the browser protocol, for both consumers.

Usage::

    python -m local_operator.browser_bridge.gen_ts
    python -m local_operator.browser_bridge.gen_ts --check
    python -m local_operator.browser_bridge.gen_ts --bundle-out <dir>   # local-operator-ui

Two targets, one source of truth:

* the **extension** target — ``extension/src/protocol.gen.ts`` — the wire
  declarations the released Chromium extension compiles against. Its bytes are
  deliberately unchanged by the addition of the second target, because a
  generated file that churns for an unrelated reason makes ``--check`` useless.
* the **bundle** target — ``extension/ui-vendor/`` — the desktop app's copy of
  the same wire declarations PLUS the host-free driver policy modules the two
  hosts share. ``local-operator-ui``'s ``scripts/sync-vendored.mjs`` is the only
  thing that writes them into that repo, and it drives this generator with
  ``--bundle-out``; the committed copy here is what makes ``--check`` a real gate
  inside THIS repo, where no UI checkout exists.

Why the header stamps an INPUT HASH and never a git SHA: every commit in this
repository would otherwise invalidate the stamp, so ``--check`` would be red on
commits that touched nothing this generator reads — and a gate that cries wolf on
every unrelated commit gets deleted. The hash covers exactly the inputs
(``protocol.py``, this file, and each ``driver/*.ts``), so it changes when — and
only when — the generated content could have changed.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

from local_operator.browser_bridge.protocol import (
    EXPECTED_EXTENSION_VERSION,
    EXTENSION_UPDATE_NOTE,
    METHODS,
    ORIGIN_PROMPT_TIMEOUT_MS,
    PROTO_VERSION,
    ErrorCode,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
TARGET = REPO_ROOT / "extension" / "src" / "protocol.gen.ts"

#: Where the host-free driver policy modules live, once the `refactor(browser-
#: driver)` move has landed. Read, never written: this generator copies them, and
#: their absence is reported rather than invented (see `driver_module_names`).
DRIVER_DIR = REPO_ROOT / "extension" / "src" / "driver"

#: The vendored bundle's committed home IN THIS REPO. A directory, unlike the
#: extension target, because the bundle is more than one file. Deliberately not
#: the UI repo's own `src/main/browser/vendor/`: that tree has exactly one writer
#: (`sync-vendored.mjs`) and a second one would make its provenance manifest a
#: lie.
BUNDLE_TARGET = REPO_ROOT / "extension" / "ui-vendor"

#: The bundle's file names, in emission order. `protocol.gen.ts` keeps that name
#: in both trees so a reader of either repo sees the same artefact.
BUNDLE_PROTOCOL_NAME = "protocol.gen.ts"


def _protocol_body() -> str:
    """The wire declarations themselves, with no generator header.

    Split out of :func:`render` so both targets emit identical declarations from
    one place: two renderers would be two chances for the extension and the app
    to disagree about a method name.
    """
    error_values = "\n".join(f"  {item.name} = {item.value!r}," for item in ErrorCode)
    methods = " | ".join(repr(item) for item in METHODS)
    return f"""export const PROTO_VERSION = {PROTO_VERSION} as const;
// The extension version this runtime was developed and released alongside.
// ADVISORY ONLY: nothing is refused for being older (`MIN_SUPPORTED_PROTO` is
// the compatibility floor). Generated so the popup's update line and the
// daemon's agree by construction.
export const EXPECTED_EXTENSION_VERSION = {EXPECTED_EXTENSION_VERSION!r} as const;
// The ONE spelling of the "a newer extension exists" advisory. `{{have}}` and
// `{{want}}` are the reported and the expected extension versions; the
// contingency is on the Chrome Web Store because nothing here can know what the
// store currently offers. Nothing is blocked by an older extension — the copy
// says so, and must keep saying so.
export const EXTENSION_UPDATE_NOTE = {EXTENSION_UPDATE_NOTE!r};
// How long the approval popup waits for a human origin decision before
// auto-denying. Generated so the daemon's prompt window (this + margin) and
// the session client's HTTP timeout can never drift below it (finding A3).
export const ORIGIN_PROMPT_TIMEOUT_MS = {ORIGIN_PROMPT_TIMEOUT_MS} as const;

export enum ErrorCode {{
{error_values}
}}

export type Method = {methods};
// One buffered console/runtime log line, as `logs` returns it (newest last).
// `level` is normalized to the error/warning/info/log vocabulary the tool
// filters on; `source` distinguishes a page console call from an uncaught
// exception ('console' | 'exception' | 'log-entry').
export interface LogEntry {{
  level: string; text: string; source: string; url: string; line: number; timestamp: number;
}}
// What `scroll` reports back so the agent knows where the viewport landed and
// whether more content remains past it (so it can stop paging at the end).
export interface ScrollResult {{
  scrollX: number; scrollY: number; moreBelow: boolean; moreRight: boolean;
  url: string; title: string;
}}
export interface Request {{ id: string; method: Method; params: Record<string, unknown>; }}
export interface ErrorDetail {{ code: ErrorCode; message: string; data: Record<string, unknown>; }}
export type Response =
  | {{ id: string; ok: true; result: Record<string, unknown> }}
  | {{ id: string; ok: false; error: ErrorDetail }};
export interface Hello {{
  event: 'hello'; proto: number; token: string; extension_version: string; browser: string;
}}
export interface HelloAck {{
  event: 'hello_ack'; proto: number; paired: boolean;
  // ADDITIVE: a daemon released before multi-identity simply does not send
  // these, so an absent `role` MUST be read as 'driver' (behave exactly as
  // before). `authorized_count` defaults to 1 for the same reason. Nothing is
  // added to `Hello` instead: the daemon validates it with extra="forbid", so
  // a new field THERE would be closed 4001 by every already-released daemon.
  role?: 'driver' | 'standby';
  authorized_count?: number;
}}
// Daemon -> extension: this link's role CHANGED while it stayed connected
// (a failover, or `lop browser drive`). A separate EVENT rather than a field on
// an existing model, so a released daemon that does not know it ignores the
// frame instead of closing the socket.
export interface Role {{ event: 'role'; role: 'driver' | 'standby'; }}
export interface PairRequest {{ event: 'pair'; code: string; }}
export interface PairResult {{ event: 'pair_result'; ok: boolean; token: string; message: string; }}
export interface Ping {{ event: 'ping'; }}
export interface Pong {{ event: 'pong'; }}
export interface TabClosed {{ event: 'tab_closed'; tab: string; }}
export interface TabUpdate {{ event: 'tab_update'; tab: string; url: string; title: string; }}
export interface AwaitingOrigin {{ event: 'awaiting_origin'; id: string; origin: string; }}
export interface AwaitingOriginCleared {{ event: 'awaiting_origin_cleared'; id: string; }}
export interface Unpair {{ event: 'unpair'; }}
export interface OriginDecision {{
  event: 'origin_decision'; origin: string; decision: 'once' | 'site' | 'domain' | 'deny';
}}
export type ExtensionEvent =
  | Hello | PairRequest | Pong | TabClosed | TabUpdate | AwaitingOrigin | AwaitingOriginCleared
  | Unpair | OriginDecision;
export type DaemonMessage = HelloAck | PairResult | Ping | Request | Role;
"""


def render() -> str:
    """The EXTENSION target's exact bytes. Unchanged by the bundle's existence."""
    return (
        "// GENERATED by python -m local_operator.browser_bridge.gen_ts. Do not edit.\n"
        + _protocol_body()
    )


def driver_module_names() -> list[str]:
    """The host-free driver modules present in `extension/src/driver/`, sorted.

    Discovered rather than hardcoded: the files are MOVED there by a separate
    change, and a hardcoded list would either fail that change's CI or silently
    emit a stale subset. An empty list is a legitimate state — it means the move
    has not landed — and `main` says so on stderr rather than pretending the
    bundle is complete.
    """
    if not DRIVER_DIR.is_dir():
        return []
    return sorted(path.name for path in DRIVER_DIR.glob("*.ts"))


def _input_paths() -> list[Path]:
    """Exactly what the generated bytes depend on, in a stable order."""
    paths = [
        Path(__file__).resolve().parent / "protocol.py",
        Path(__file__).resolve(),
    ]
    paths.extend(DRIVER_DIR / name for name in driver_module_names())
    return paths


def inputs_sha256() -> str:
    """sha256 over the generator's INPUTS — never over a git SHA.

    Each path's name is hashed with its bytes, so a rename with identical content
    still moves the stamp (a rename changes the emitted module path, and the
    stamp must be able to say the bundle is stale).
    """
    digest = hashlib.sha256()
    for path in _input_paths():
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _bundle_header(inputs: str, *, relative: str, kind: str) -> str:
    """The bundle header: provenance, the input stamp, and NOTHING machine-specific.

    The header names the file by its path WITHIN the bundle, never by absolute
    path: the bytes must be identical wherever the bundle is written, or the UI
    repo's vendored copy could not be diffed against this repo's committed one,
    and `--check` would fail on any other machine's checkout root.
    """
    names = driver_module_names()
    modules = ", ".join(names) if names else "(none — extension/src/driver/ is not present yet)"
    return (
        "// GENERATED by python -m local_operator.browser_bridge.gen_ts. Do not edit.\n"
        f"// Vendored copy for local-operator-ui: {relative} ({kind}).\n"
        "// Source of truth: local-operator "
        "local_operator/browser_bridge/protocol.py + gen_ts.py"
        + (f" + extension/src/driver/*.ts ({len(names)} modules: {modules})" if names else "")
        + "\n"
        f"// PROTO_VERSION: {PROTO_VERSION}\n"
        f"// Inputs sha256: {inputs}\n"
        "// An INPUT hash, never a git SHA: a stamp over commits would go red on every\n"
        "// commit that touched nothing this generator reads, and a gate that cries wolf\n"
        "// gets deleted. Regenerate with `python -m local_operator.browser_bridge.gen_ts`;\n"
        "// `--bundle-out DIR` is the same command pointed at another tree, which is how\n"
        "// local-operator-ui's sync script generates its own checkout.\n"
    )


def render_bundle() -> dict[str, str]:
    """The UI host's bundle: relative path -> exact file contents.

    Contains the wire declarations plus a verbatim copy of each host-free driver
    module. NOT a copy of the chrome-coupled extension modules: those differ in
    SEMANTICS under Electron (a different debugger attach model, different
    navigation events), so they are re-implemented against Electron rather than
    adapted, and pretending otherwise would produce a bundle nobody could use.

    Takes no destination on purpose — see :func:`_bundle_header`.
    """
    inputs = inputs_sha256()
    files = {
        BUNDLE_PROTOCOL_NAME: _bundle_header(
            inputs, relative=BUNDLE_PROTOCOL_NAME, kind="wire declarations"
        )
        + _protocol_body()
    }
    for name in driver_module_names():
        files[f"driver/{name}"] = _bundle_header(
            inputs,
            relative=f"driver/{name}",
            kind="host-free shared policy (chrome.*-free by construction)",
        ) + (DRIVER_DIR / name).read_text(encoding="utf-8")
    return files


def _report_missing_driver_move() -> None:
    if not driver_module_names():
        print(
            f"note: {DRIVER_DIR} does not exist yet, so the bundle carries the wire "
            "declarations only. The host-free driver modules move there in a separate "
            "change; regenerate this bundle when it lands.",
            file=sys.stderr,
        )


def _check_target(current: str, expected: str, label: str) -> bool:
    if current == expected:
        return True
    print(f"{label} is stale; run python -m local_operator.browser_bridge.gen_ts", file=sys.stderr)
    return False


def _check_bundle(bundle_dir: Path) -> bool:
    """Verify a bundle tree byte-exactly, and that it lists nothing extra.

    The extra-file rule is the same one `check-vendored.mjs` applies on the UI
    side: a hand-added file in a generated tree is the failure that has no other
    detector, because every generated file it should contain still matches.
    """
    expected = render_bundle()
    ok = True
    for relative, content in expected.items():
        path = bundle_dir / relative
        try:
            current = path.read_text(encoding="utf-8")
        except OSError:
            current = ""
        ok &= _check_target(current, content, str(path))
    if bundle_dir.is_dir():
        for path in sorted(bundle_dir.rglob("*")):
            if path.is_file() and path.relative_to(bundle_dir).as_posix() not in expected:
                print(
                    f"{path} is not generated by this tool; remove it (a generated tree "
                    "lists only what it emits, so a hand-added file is a silent fork)",
                    file=sys.stderr,
                )
                ok = False
    return ok


def write_bundle(bundle_dir: Path) -> list[Path]:
    written: list[Path] = []
    for relative, content in render_bundle().items():
        path = bundle_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        written.append(path)
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument(
        "--bundle-out",
        default=None,
        help=(
            "Write/check the UI host's bundle at this directory instead of "
            f"{BUNDLE_TARGET} (local-operator-ui's sync script passes its own tree)."
        ),
    )
    parser.add_argument(
        "--no-bundle",
        action="store_true",
        help="Touch only the extension target (the historical single-target behaviour).",
    )
    args = parser.parse_args(argv)
    bundle_dir = Path(args.bundle_out).resolve() if args.bundle_out else BUNDLE_TARGET
    bundle_enabled = not args.no_bundle

    if args.check:
        current = TARGET.read_text(encoding="utf-8") if TARGET.exists() else ""
        ok = _check_target(current, render(), str(TARGET))
        if bundle_enabled:
            _report_missing_driver_move()
            ok &= _check_bundle(bundle_dir)
        return 0 if ok else 1

    TARGET.parent.mkdir(parents=True, exist_ok=True)
    TARGET.write_text(render(), encoding="utf-8")
    print(TARGET)
    if bundle_enabled:
        _report_missing_driver_move()
        for path in write_bundle(bundle_dir):
            print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
