"""``lop operator`` verbs. Imported only after an ``operator`` verb is dispatched.

THE COPY RULES THESE HANDLERS FOLLOW, all of them from the design round and the
acceptance matrix:

* ``init`` reports the LEVEL it achieved. A setup step that printed "operator key
  created" without saying whether a signature costs a human gesture would be
  exactly the overclaim §2.2 exists to prevent, so the level line is the verb's
  main output rather than a footnote.
* ``sign`` prints machine-readable JSON on stdout and every human word on
  stderr, because its consumer is the TUI's attached pane or the desktop backend
  (and, in stage D, the relay) — a banner line on stdout would corrupt the one
  value crossing that pipe, the same failure ``lop secret get`` documents.
* nothing here ever prints key material: the private half never leaves the
  backend, and the anchor holds public data by construction.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys

from local_operator.operator import (
    LEVEL_ANCHOR_UNPINNED,
    LEVEL_OPERATOR_FILE_ONLY,
    LEVEL_OPERATOR_PRESENCE,
    LEVEL_SPAWN_ONLY,
    anchor_bytes,
    anchor_path,
    create_key,
    install_commands,
    load_anchor,
    operator_authority_report,
    sign_challenge,
    staging_path,
)
from local_operator.operator.keychain import KeyBackendError
from local_operator.operator.sign import anchor_for_handle, describe_level
from local_operator.paths import config_dir


def dispatch(args: argparse.Namespace) -> int:
    command = getattr(args, "operator_command", None)
    if command == "init":
        return _init(args)
    if command == "install":
        return _install(args)
    if command == "trust":
        return _trust()
    if command == "status":
        return _status()
    if command == "sign":
        return _sign(args)
    print("usage: lop operator {init|trust|install|sign|status}", file=sys.stderr)
    return 2


def _is_privileged() -> bool:
    """Whether this process may already write the anchor's root-owned path.

    POSIX answers with ``os.geteuid``; Windows has no such attribute (reading it
    raises ``AttributeError``, which the probe battery counts as a fatal
    unguarded POSIX attribute), and there is no cheap, dependency-free way to ask
    whether the token is elevated — so Windows answers ``False`` and takes the
    print-the-command path. That is the honest outcome: a Windows operator runs
    the PowerShell step from an elevated shell, and a `lop` process that guessed
    "elevated" would write a non-administrator-owned file into ``%PROGRAMDATA%``
    and call it an anchor.
    """
    geteuid = getattr(os, "geteuid", None)
    return bool(geteuid is not None and geteuid() == 0)


def _init(args: argparse.Namespace) -> int:
    root = config_dir()
    try:
        handle = create_key(config_root=root, preference=args.backend)
    except KeyBackendError as exc:
        print(f"could not create the operator key: {exc}", file=sys.stderr)
        return 1
    anchor = anchor_for_handle(handle, label=args.label)
    staged = staging_path(root)
    staged.parent.mkdir(parents=True, exist_ok=True)
    os.chmod(staged.parent, 0o700)
    staged.write_bytes(anchor_bytes(anchor))
    os.chmod(staged, 0o600)

    print(f"operator key created in the {handle.backend} store")
    print(f"  key id : {handle.key_id}")
    print(f"  level  : {describe_level(handle)}")
    print(f"  staged : {staged}")
    print()
    print("The runtime trusts ONLY the root-owned anchor. Install it with:")
    print("  lop operator install")
    if handle.backend == "file-only":
        print()
        print(
            "WARNING: this host has no presence store, so loosening is authorised by a "
            "key any process running as you can read. That is reported as a lower level "
            "and is NOT a boundary.",
            file=sys.stderr,
        )
    return 0


def _install(args: argparse.Namespace) -> int:
    root = config_dir()
    staged = staging_path(root)
    target = anchor_path()
    if not staged.exists():
        print(f"nothing staged at {staged}; run `lop operator init` first", file=sys.stderr)
        return 1
    command = install_commands(staged, target)
    if args.print_only:
        print(" && ".join(" ".join(argv) for argv in command))
        return 0
    if _is_privileged():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(staged, target)
        os.chmod(target, 0o644)
    else:
        if shutil.which("sudo") is None:
            print(
                "installing the anchor needs administrator rights. Run this by hand:\n  "
                + "\n  ".join(" ".join(argv) for argv in command),
                file=sys.stderr,
            )
            return 1
        print(
            "installing the anchor as root — this is the ONE privileged step, and sudo "
            "will ask for your password:",
            file=sys.stderr,
        )
        for argv in command:
            result = subprocess.run(argv, check=False)
            if result.returncode != 0:
                return result.returncode
    loaded = load_anchor()
    print(f"anchor installed at {loaded.path} (trusted: {loaded.usable})")
    if not loaded.usable:
        print(f"  the runtime will NOT trust it: {loaded.reason}", file=sys.stderr)
        return 1
    return 0


def _trust() -> int:
    loaded = load_anchor()
    report = operator_authority_report()
    print(f"anchor    : {loaded.path}")
    print(f"installed : {loaded.exists}")
    print(f"trusted   : {loaded.usable}")
    if loaded.anchor:
        print(f"key id    : {loaded.anchor.key_id}")
        print(f"backend   : {loaded.anchor.backend} (presence: {loaded.anchor.presence})")
    if not loaded.usable:
        print(f"reason    : {loaded.reason}", file=sys.stderr)
    print(f"level     : {report['level']}")
    return 0 if loaded.usable else 1


def _status() -> int:
    report = operator_authority_report()
    staged = staging_path(config_dir())
    print(f"level                  : {report['level']}")
    print(f"anchor path            : {report['anchor_path']}")
    print(f"anchor installed       : {report['anchor_installed']}")
    print(f"anchor root-owned      : {report['anchor_root_owned']}")
    print(f"private-half backend   : {report['backend'] or '(none)'}")
    print(f"presence per signature : {report['presence_enforced_by_os']}")
    print(f"spawn-capable guarantee: {report['capability_guarantee']}")
    print(f"reason                 : {report['reason']}")
    # THE ONE STEP THAT IS STILL PENDING, named rather than left for the reader to
    # infer from a level that looks lower than what `init` just printed. The two
    # lines are about different things on purpose: `init` reports the KEY it
    # created, this reports what the RUNTIME will honour, and between them sits
    # exactly one privileged command.
    if staged.exists() and not report["anchor_installed"]:
        print(f"staged anchor          : {staged} (run `lop operator install` to trust it)")
    if report["level"] == LEVEL_OPERATOR_PRESENCE:
        print("loosening: authorised by a signature that costs a human gesture")
    elif report["level"] == LEVEL_OPERATOR_FILE_ONLY:
        print("loosening: authorised by a key ANY process running as you can read — not a boundary")
    elif report["level"] == LEVEL_ANCHOR_UNPINNED:
        print("loosening: refused — a file sits where the anchor belongs and is not root-owned")
    elif report["level"] == LEVEL_SPAWN_ONLY:
        print("loosening: only the process that started the session may loosen it")
    return 0


def _sign(args: argparse.Namespace) -> int:
    root = config_dir()
    try:
        signature = sign_challenge(
            challenge=args.challenge,
            purpose=args.purpose,
            config_root=root,
            session_id=args.session,
            request_id=args.request_id,
        )
    except KeyBackendError as exc:
        print(f"could not sign: {exc}", file=sys.stderr)
        return 1
    # stdout carries the value and nothing else: see the module docstring.
    sys.stdout.write(json.dumps(signature.as_json(), separators=(",", ":")) + "\n")
    return 0
