"""``lop vault connect|status|disconnect``: the person's side of agent access.

WHY A COMMAND AND NOT A TOOL VERB. Granting an agent access is a HUMAN act: the
person compares a fingerprint and approves with their own sign-in on a trusted
page (VAULT.md §4, §6.20). The agent never starts it; the ``secret`` tool tells
the agent to ask its person to run ``lop vault connect``.

``connect``:

1. Forces a refresh of the Team record and the Vault record (the gate,
   ``eligibility.py``). A closed gate prints the one sentence that names why
   (signed out, no Team plan, no Team Bridge, not approved, not available yet)
   and stops. In production today route 1 reports ``delegations_available:
   false``, so this prints "Agent access isn't available yet." and exits — the
   dark state, never an error loop.
2. Loads or creates the delegation key for the owner's endpoint.
3. Prints the key fingerprint as 4 lines of 4 groups, the same grouping the
   approval page shows, with the instruction to compare them.
4. Opens ``/dashboard/vaults/delegate?tenant=&ep=&pk=`` (VAULT.md §4: the link
   carries identifiers and the public key only, never a secret).
5. Reads route 29 every :data:`POLL_SECONDS` until a ``live`` row matches this
   endpoint and fingerprint exactly, caches it, and prints its expiry. A
   ``503`` stops the wait with the same "not available yet" sentence.

``disconnect`` deletes the key and forgets the grant, then points at the
console list: revocation is human-asserted, so the harness cannot revoke a
grant itself, and a grant without its key can never be used anyway.

The help is LISTED only when the gate is open (operator constraint: advertised
only with access), with the hide idiom ``hub_org.cli`` uses. Copy is keyed
(``cli.vault``/``cli.vault_help``); the fingerprint hex, ids and URLs are data.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlencode

from local_operator.i18n.keys import cli_vault as _v
from local_operator.i18n.keys import cli_vault_help as _vh

#: The console origin the person approves on. A constant, as the tunnel and
#: recovery surfaces spell theirs: the console is not configurable per machine,
#: and a configurable approval origin would be a phishing lever.
CONSOLE_ORIGIN = "https://console.radienthq.com"
DELEGATE_PATH = "/dashboard/vaults/delegate"
AGENTS_PATH = "/dashboard/vaults/agents"

#: Design §4.4: every 5 s for up to 10 min. A grant takes a person a minute or
#: two; polling faster buys nothing and spends the account's rate limit.
POLL_SECONDS = 5.0
DEFAULT_TIMEOUT_SECONDS = 600
#: Consecutive transient route-29 failures tolerated before giving up the wait.
MAX_POLL_ERRORS = 3

EXIT_OK = 0
EXIT_NOT_CONNECTED = 1
EXIT_INTERRUPTED = 130


def _t(msg: Any) -> str:
    return msg.envelope()["text"]


def _root() -> Path:
    from local_operator.paths import config_dir

    return Path(config_dir())


def _gate_open(root: Path) -> bool:
    from local_operator.vault_agent.eligibility import is_open

    try:
        return is_open(root)
    except Exception:  # noqa: BLE001 - parser construction never fails over the gate
        return False


def add_parser(subparsers: Any, *, parent_parser: Any, help_locale: str | None = None) -> None:
    """Register ``vault``; LISTED in help only when the cached gate is open."""

    def h(msg: Any) -> str:
        return msg.envelope(locale=help_locale)["text"] if help_locale else _t(msg)

    try:
        shown = _gate_open(_root())
    except Exception:  # noqa: BLE001
        shown = False
    # No ``help=`` hides a subparser row (``help=SUPPRESS`` would print the
    # sentinel); see ``resume-click`` in ``cli.py``.
    kwargs: dict[str, Any] = {"help": h(_vh.vault())} if shown else {}
    vault = subparsers.add_parser("vault", parents=[parent_parser], **kwargs)
    actions = vault.add_subparsers(dest="vault_command")
    connect = actions.add_parser("connect", help=h(_vh.connect()), parents=[parent_parser])
    connect.add_argument("--no-open", action="store_true", help=h(_vh.no_open()))
    connect.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT_SECONDS,
        metavar="SECONDS",
        help=h(_vh.timeout()),
    )
    status = actions.add_parser("status", help=h(_vh.status()), parents=[parent_parser])
    status.add_argument("--json", action="store_true", help=h(_vh.json()))
    actions.add_parser("disconnect", help=h(_vh.disconnect()), parents=[parent_parser])


# -- shared pieces -----------------------------------------------------------------


def delegate_url(*, tenant_id: str, endpoint_id: str, public_key_b64url: str) -> str:
    """The console approval link. Identifiers and the public key, never a secret."""

    query = urlencode({"tenant": tenant_id, "ep": endpoint_id, "pk": public_key_b64url})
    return f"{CONSOLE_ORIGIN}{DELEGATE_PATH}?{query}"


def _local_time(stamp: str) -> str:
    """An RFC 3339 instant in the person's local time zone, unambiguous.

    A stamp without an offset is UTC per the contract (``parse_instant``), never
    this machine's local time.
    """

    from local_operator.vault_agent.eligibility import parse_instant

    when = parse_instant(stamp)
    if when is None:
        return stamp
    return when.astimezone().strftime("%Y-%m-%d %H:%M %Z").strip()


def _closed_reason(observation: Any) -> str:
    org = observation.tenant_name or observation.tenant_id or ""
    return {
        "signed_out": lambda: _t(_v.signed_out()),
        "not_team": lambda: _t(_v.not_team()),
        "no_endpoint": lambda: _t(_v.no_endpoint()),
        "not_eligible": lambda: _t(_v.not_eligible(org=org)),
        "frozen": lambda: _t(_v.frozen(org=org)),
        "disabled": lambda: _t(_v.disabled(org=org)),
        "not_approved": lambda: _t(_v.not_approved(org=org)),
        "unavailable": lambda: _t(_v.unavailable()),
    }.get(observation.state, lambda: _t(_v.error()))()


def _refresh(root: Path) -> Any:
    """Force both gate records; return the Vault observation."""

    from local_operator.hub_org import eligibility as hub
    from local_operator.vault_agent import eligibility

    hub.refresh(root, boundary="explicit", force=True)
    return eligibility.refresh(root, boundary="explicit", force=True)


def _link(url: str, out: Any) -> str:
    """``url`` on its own line, as one OSC 8 hyperlink when that is safe.

    Unindented and alone on its line so the terminal's own URL detection sees
    the whole link however many rows it wraps to (an indent split the click
    target at the first row). OSC 8 only on a TTY whose emulator is known to
    render it (``terminals.is_hyperlinked``): piped into a file or a script,
    the escape would be noise, so the plain URL is the fallback.
    """

    from local_operator.terminals import is_hyperlinked

    isatty = getattr(out, "isatty", None)
    try:
        linked = bool(callable(isatty) and isatty()) and is_hyperlinked()
    except Exception:  # noqa: BLE001 - the plain URL is always correct
        linked = False
    return f"\x1b]8;;{url}\x1b\\{url}\x1b]8;;\x1b\\" if linked else url


def _print_title(org: str | None, *, out: Any) -> None:
    # The organization NAME only: a bare ``tnt_`` id is accurate but tells a
    # person nothing, so when no name is known the title stands alone.
    print(_t(_v.title(org=org) if org else _v.title_bare()), file=out)


def _print_key_block(
    *,
    org: str | None,
    endpoint_id: str,
    fingerprint: str,
    out: Any,
    held: bool = False,
    compare: bool = True,
) -> None:
    """The title, the endpoint row and the fingerprint, set apart for comparing.

    The two labels are the console's own ("Agent endpoint", "Key fingerprint")
    so the person matches the two screens at a glance. The instruction comes
    BEFORE the label (people read what to do, then compare), so "Key
    fingerprint:" sits directly over its value, and blank lines set the grid
    apart. Every row stays short of 60 cells, so nothing wraps mid-identifier.
    """

    from local_operator.vault_agent.wire import fingerprint_grid

    _print_title(org, out=out)
    print(_t(_v.endpoint_row(endpoint=endpoint_id)), file=out)
    # ``held``: the gate is closed, so the key exists but is not usable now.
    # ``compare``: only ``connect`` asks for the comparison; ``status`` reports.
    if compare and not held:
        print(_t(_v.compare()), file=out)
        print(_t(_v.compare_stop()), file=out)
    print(_t(_v.fingerprint_held() if held else _v.fingerprint_label()), file=out)
    print(file=out)
    for line in fingerprint_grid(fingerprint):
        print(f"  {line}", file=out)
    print(file=out)


def _print_connected(expires_at: str, *, out: Any) -> None:
    # The state and the one fact that matters later lead the line.
    print(_t(_v.connected_until(until=_local_time(expires_at))), file=out)
    print(_t(_v.connected_scope()), file=out)


def _waiting(timeout: int) -> str:
    # Rounded UP: "up to 1 minute" for a 90 s wait would be a promise the wait
    # does not keep.
    if timeout < 60:
        return _t(_v.waiting_seconds(seconds=max(1, timeout)))
    return _t(_v.waiting(minutes=math.ceil(timeout / 60)))


# -- verbs -------------------------------------------------------------------------


def connect(
    root: Path,
    *,
    open_browser: bool = True,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
    out: Any = None,
    err: Any = None,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
    opener: Callable[[str], Any] | None = None,
    client_factory: Callable[[Path], Any] | None = None,
) -> int:
    """Refusals, timeouts and errors go to ``err`` (stderr), as ``hub_org`` and
    the tunnel CLI do; the key block, the link and success go to ``out``."""

    from local_operator.vault_agent import eligibility, keys, wire
    from local_operator.vault_agent.resolve import default_client_factory, pick_up

    out = out or sys.stdout
    err = err or sys.stderr
    observation = _refresh(root)
    gate = eligibility.gate(root)
    if gate is None:
        print(_closed_reason(observation), file=err)
        return EXIT_NOT_CONNECTED

    try:
        seed, _created = keys.load_or_create_seed(gate.endpoint_id, root)
    except Exception:  # noqa: BLE001 - a damaged row or store failure: a remedy, not a traceback
        print(_t(_v.key_unreadable()), file=err)
        return EXIT_NOT_CONNECTED
    public_key = keys.public_key_from_seed(seed)
    fingerprint = wire.key_fingerprint(public_key)
    _print_key_block(
        org=gate.tenant_name,
        endpoint_id=gate.endpoint_id,
        fingerprint=fingerprint,
        out=out,
    )
    url = delegate_url(
        tenant_id=gate.tenant_id,
        endpoint_id=gate.endpoint_id,
        public_key_b64url=wire.b64url(public_key),
    )
    opened = False
    if open_browser:
        if opener is None:
            import webbrowser

            opener = webbrowser.open
        try:
            opened = bool(opener(url))
        except Exception:  # noqa: BLE001 - the printed link is the fallback
            opened = False
    print(_t(_v.opening() if opened else _v.open_manually()), file=out)
    print(_link(url, out), file=out)

    try:
        client = (client_factory or default_client_factory)(root)
    except Exception:  # noqa: BLE001 - an unreadable sign-in reads as signed out
        client = None
    if client is None:
        print(_t(_v.signed_out()), file=err)
        return EXIT_NOT_CONNECTED
    print(_waiting(timeout), file=out, flush=True)
    deadline = clock() + max(0, timeout)
    errors = 0
    try:
        while True:
            try:
                grant = pick_up(
                    client,
                    tenant_id=gate.tenant_id,
                    endpoint_id=gate.endpoint_id,
                    fingerprint=fingerprint,
                )
                errors = 0
            except Exception as exc:  # noqa: BLE001 - classified
                status = getattr(exc, "status_code", None)
                code = getattr(exc, "code", None)
                if status == 503 or code == "vault_unavailable":
                    # The dark gate or an outage: one sentence, no loop.
                    print(_t(_v.unavailable()), file=err)
                    return EXIT_NOT_CONNECTED
                if status == 401:
                    # A refused sign-in, not a membership problem: naming the
                    # membership would send the person to an admin for nothing.
                    print(_t(_v.signed_out()), file=err)
                    return EXIT_NOT_CONNECTED
                if code == "vault_no_access" or status == 403:
                    org = gate.tenant_name or gate.tenant_id
                    print(_t(_v.no_access(org=org)), file=err)
                    return EXIT_NOT_CONNECTED
                errors += 1
                if errors >= MAX_POLL_ERRORS:
                    print(_t(_v.error()), file=err)
                    return EXIT_NOT_CONNECTED
                grant = None
            if grant is not None:
                eligibility.write_grant(root, grant)
                _print_connected(grant.expires_at, out=out)
                return EXIT_OK
            if clock() >= deadline:
                print(_t(_v.timed_out()), file=err)
                return EXIT_NOT_CONNECTED
            sleep(POLL_SECONDS)
    except KeyboardInterrupt:
        print(_t(_v.interrupted()), file=err)
        return EXIT_INTERRUPTED


def status(root: Path, *, as_json: bool = False, out: Any = None, err: Any = None) -> int:
    """Report, never act: exit 0 in every state, as ``lop tunnel status`` does.

    A script reads ``connected`` from ``--json`` instead of an exit code: it is
    true only when the gate is open AND a matching unexpired grant is cached,
    the same test the human "Connected until" line uses. A CLOSED gate never
    says "Connected" (a cached grant the closed gate will not let
    ``vault_resolve`` use is not access) and never advises ``connect``, which
    would only refuse on the reason printed; that reason is the LAST line, the
    one people act on.
    """

    from local_operator.team_bridge.store import Store
    from local_operator.vault_agent import eligibility, keys

    out = out or sys.stdout
    err = err or sys.stderr
    observation = _refresh(root)
    gate = eligibility.gate(root)
    owner = gate or Store(root).active_owner_readonly()
    unreadable = False
    try:
        seed = keys.load_seed(owner.endpoint_id, root) if owner is not None else None
    except Exception:  # noqa: BLE001 - a damaged row or store failure: a remedy, not a traceback
        seed, unreadable = None, True
    fingerprint = keys.fingerprint_for_seed(seed) if seed is not None else None
    grant = (
        eligibility.read_grant(
            root,
            tenant_id=owner.tenant_id,
            endpoint_id=owner.endpoint_id,
            fingerprint=fingerprint,
        )
        if gate is not None and owner is not None and fingerprint is not None
        else None
    )
    org = getattr(gate, "tenant_name", None) or observation.tenant_name
    if as_json:
        print(
            json.dumps(
                {
                    "available": gate is not None,
                    "connected": grant is not None,
                    "state": "open" if gate is not None else observation.state,
                    "tenant_id": getattr(owner, "tenant_id", None),
                    "endpoint_id": getattr(owner, "endpoint_id", None),
                    "key_fingerprint": fingerprint,
                    "delegation_id": grant.delegation_id if grant else None,
                    "expires_at": grant.expires_at if grant else None,
                },
                indent=2,
            ),
            file=out,
        )
        return EXIT_OK
    if unreadable:
        print(_t(_v.key_unreadable()), file=err)
        return EXIT_OK
    if gate is None:
        if owner is not None and fingerprint is not None:
            _print_key_block(
                org=org, endpoint_id=owner.endpoint_id, fingerprint=fingerprint, out=out, held=True
            )
        print(_closed_reason(observation), file=out)
        return EXIT_OK
    if fingerprint is None:
        _print_title(org, out=out)
        print(_t(_v.status_no_key()), file=out)
        return EXIT_OK
    _print_key_block(
        org=org, endpoint_id=gate.endpoint_id, fingerprint=fingerprint, out=out, compare=False
    )
    if grant is not None:
        _print_connected(grant.expires_at, out=out)
    else:
        print(_t(_v.status_not_connected()), file=out)
    return EXIT_OK


def disconnect(root: Path, *, out: Any = None, err: Any = None) -> int:
    """Delete EVERY delegation key on this machine and forget the cached grant.

    Every key, not the current endpoint's: an earlier enrolment's key is
    otherwise unremovable (``keys.delete_all_seeds``). The count is reported
    truthfully, including zero.
    """

    from local_operator.vault_agent import eligibility, keys

    out = out or sys.stdout
    err = err or sys.stderr
    try:
        removed = keys.delete_all_seeds(root)
    except Exception:  # noqa: BLE001 - a store or broker failure: say so, no traceback
        print(_t(_v.store_error()), file=err)
        return EXIT_NOT_CONNECTED
    eligibility.clear_grant(root)
    print(_t(_v.disconnected(count=removed) if removed else _v.nothing_to_remove()), file=out)
    print(_t(_v.revoke_hint()), file=out)
    print(_link(f"{CONSOLE_ORIGIN}{AGENTS_PATH}", out), file=out)
    return EXIT_OK


def main(args: argparse.Namespace) -> int:
    root = _root()
    command = getattr(args, "vault_command", None) or "status"
    if command == "connect":
        return connect(root, open_browser=not args.no_open, timeout=args.timeout)
    if command == "disconnect":
        return disconnect(root)
    return status(root, as_json=bool(getattr(args, "json", False)))
