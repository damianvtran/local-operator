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
import sys
import time
from datetime import datetime
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
    """An RFC 3339 instant in the person's local time zone, unambiguous."""

    try:
        when = datetime.fromisoformat(stamp.replace("Z", "+00:00")).astimezone()
    except ValueError:
        return stamp
    return when.strftime("%Y-%m-%d %H:%M %Z").strip()


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


def _print_fingerprint(fingerprint: str, *, out: Any) -> None:
    from local_operator.vault_agent.wire import fingerprint_grid

    for line in fingerprint_grid(fingerprint):
        print(f"  {line}", file=out)


# -- verbs -------------------------------------------------------------------------


def connect(
    root: Path,
    *,
    open_browser: bool = True,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
    out: Any = None,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
    opener: Callable[[str], Any] | None = None,
    client_factory: Callable[[Path], Any] | None = None,
) -> int:
    from local_operator.vault_agent import eligibility, keys, wire
    from local_operator.vault_agent.resolve import default_client_factory, pick_up

    out = out or sys.stdout
    observation = _refresh(root)
    gate = eligibility.gate(root)
    if gate is None:
        print(_closed_reason(observation), file=out)
        return EXIT_NOT_CONNECTED

    seed, _created = keys.load_or_create_seed(gate.endpoint_id, root)
    public_key = keys.public_key_from_seed(seed)
    fingerprint = wire.key_fingerprint(public_key)
    print(_t(_v.key_heading(endpoint=gate.endpoint_id, org=gate.tenant_name)), file=out)
    _print_fingerprint(fingerprint, out=out)
    print(_t(_v.compare()), file=out)
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
    print(f"  {url}", file=out)

    client = (client_factory or default_client_factory)(root)
    if client is None:
        print(_t(_v.signed_out()), file=out)
        return EXIT_NOT_CONNECTED
    print(_t(_v.waiting(minutes=max(1, round(timeout / 60)))), file=out, flush=True)
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
                    print(_t(_v.unavailable()), file=out)
                    return EXIT_NOT_CONNECTED
                if code == "vault_no_access" or status in (401, 403):
                    print(_t(_v.no_access(org=gate.tenant_name)), file=out)
                    return EXIT_NOT_CONNECTED
                errors += 1
                if errors >= MAX_POLL_ERRORS:
                    print(_t(_v.error()), file=out)
                    return EXIT_NOT_CONNECTED
                grant = None
            if grant is not None:
                eligibility.write_grant(root, grant)
                print(_t(_v.connected_until(until=_local_time(grant.expires_at))), file=out)
                return EXIT_OK
            if clock() >= deadline:
                print(_t(_v.timed_out()), file=out)
                return EXIT_NOT_CONNECTED
            sleep(POLL_SECONDS)
    except KeyboardInterrupt:
        print(_t(_v.interrupted()), file=out)
        return EXIT_INTERRUPTED


def status(root: Path, *, as_json: bool = False, out: Any = None) -> int:
    from local_operator.team_bridge.store import Store
    from local_operator.vault_agent import eligibility, keys

    out = out or sys.stdout
    observation = _refresh(root)
    gate = eligibility.gate(root)
    owner = gate or Store(root).active_owner_readonly()
    seed = keys.load_seed(owner.endpoint_id, root) if owner is not None else None
    fingerprint = keys.fingerprint_for_seed(seed) if seed is not None else None
    grant = (
        eligibility.read_grant(
            root,
            tenant_id=owner.tenant_id,
            endpoint_id=owner.endpoint_id,
            fingerprint=fingerprint,
        )
        if owner is not None and fingerprint is not None
        else None
    )
    if as_json:
        print(
            json.dumps(
                {
                    "available": gate is not None,
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
    if gate is None:
        print(_closed_reason(observation), file=out)
        if fingerprint is None:
            # Nothing to show, and "run lop vault connect" would only fail on
            # the reason just printed.
            return EXIT_OK
    else:
        print(_t(_v.status_open(org=gate.tenant_name)), file=out)
    if fingerprint is None:
        print(_t(_v.status_no_key()), file=out)
        return EXIT_OK
    print(_t(_v.key_heading(endpoint=owner.endpoint_id, org=getattr(owner, "tenant_name", owner.tenant_id))), file=out)
    _print_fingerprint(fingerprint, out=out)
    if grant is not None:
        print(_t(_v.connected_until(until=_local_time(grant.expires_at))), file=out)
    else:
        print(_t(_v.status_not_connected()), file=out)
    return EXIT_OK


def disconnect(root: Path, *, out: Any = None) -> int:
    from local_operator.team_bridge.store import Store
    from local_operator.vault_agent import eligibility, keys

    out = out or sys.stdout
    owner = Store(root).active_owner_readonly()
    removed = keys.delete_seed(owner.endpoint_id, root) if owner is not None else False
    eligibility.clear_grant(root)
    print(_t(_v.disconnected() if removed else _v.nothing_to_remove()), file=out)
    print(_t(_v.revoke_hint(url=f"{CONSOLE_ORIGIN}{AGENTS_PATH}")), file=out)
    return EXIT_OK


def main(args: argparse.Namespace) -> int:
    root = _root()
    command = getattr(args, "vault_command", None) or "status"
    if command == "connect":
        return connect(root, open_browser=not args.no_open, timeout=args.timeout)
    if command == "disconnect":
        return disconnect(root)
    return status(root, as_json=bool(getattr(args, "json", False)))
