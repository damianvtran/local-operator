"""``lop hub org [list|pull]``: the organization catalogue on the command line.

WHY A MODULE, NOT MORE ``cli.py``. The parser block, the two handlers and the
helpers shared with ``skills/agents/teams pull`` (org name resolution) and
``teams search`` (org rows) form one feature with one gate. Keeping them here
gives ``cli.py`` a few call sites rather than another 400 lines, and the tests
exercise the handlers directly.

THE GATE (operator constraint, verbatim: "that private hub is only available on
Radient Team so that shouldn't be advertised if they aren't logged in with
Radient or their Radient account is not part of a team"). The help is built
from ONE cached, network-free read, ``eligibility.is_eligible``. When the gate
is closed, the subcommand is registered WITHOUT ``help=``: the argparse hide
idiom for a subparser (see ``resume-click`` in ``cli.py``), because
``help=SUPPRESS`` on a subparser prints the sentinel. An ineligible invocation
does one forced refresh (the record may simply be stale) and, if the gate is
still closed, prints the single direct sentence and exits 1. There is no
up-sell beyond it.

Copy is keyed (``cli.hub_org_help`` for argparse help, ``cli.hub_org`` for
output). Item names, ids, kinds and outcome tokens are machine vocabulary and
stay as data.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from local_operator.i18n.keys import cli_hub_org as _ho
from local_operator.i18n.keys import cli_hub_org_help as _hh

_KIND_CHOICES = ("agent", "team", "skill")


def _t(msg: Any) -> str:
    return msg.envelope()["text"]


def _eligible(config_dir: Path) -> bool:
    from local_operator.hub_org.eligibility import is_eligible

    try:
        return is_eligible(config_dir)
    except Exception:  # noqa: BLE001 - a gate read never breaks parser construction
        return False


def add_parser(subparsers: Any, *, parent_parser: Any, help_locale: str, config_dir: Path) -> None:
    """Register ``hub org``; listed in help only when the cached gate is open."""

    def h(msg: Any) -> str:
        return msg.envelope(locale=help_locale)["text"]

    shown = _eligible(config_dir)
    # A hidden SUBPARSER takes no ``help=`` at all (``help=SUPPRESS`` renders the
    # sentinel verbatim on a subparser). Its ARGUMENTS keep their help so a user
    # who types `lop hub org pull --help` while eligible sees everything.
    kwargs: dict[str, Any] = {"help": h(_hh.org())} if shown else {}
    org = subparsers.add_parser("org", parents=[parent_parser], **kwargs)
    if not shown:
        # Omitting ``help=`` keeps the row out of the command list, but argparse
        # still spells every choice in the ``{a,b,...}`` usage metavar, so
        # `lop hub --help` would print `{status,links,link,org}` to a signed-out
        # user. Pin the metavar to the visible choices.
        visible = [name for name in subparsers.choices if name != "org"]
        subparsers.metavar = "{" + ",".join(visible) + "}"
    org_sub = org.add_subparsers(dest="hub_org_command")

    def common(p: Any) -> None:
        p.add_argument("--org", dest="org", default=None, help=h(_hh.org_flag()))
        p.add_argument(
            "--kind",
            action="append",
            choices=_KIND_CHOICES,
            default=None,
            help=h(_hh.kind()),
        )
        p.add_argument("--json", action="store_true", help=h(_hh.json()))

    # Bare `lop hub org` == `list`, so the flags also live on the group parser.
    common(org)
    common(org_sub.add_parser("list", help=h(_hh.list()), parents=[parent_parser]))
    pull = org_sub.add_parser("pull", help=h(_hh.pull()), parents=[parent_parser])
    common(pull)
    pull.add_argument("--name", action="append", default=None, help=h(_hh.name()))
    pull.add_argument(
        "--on-conflict",
        choices=("adopt-link", "adopt", "reconcile", "skip"),
        default="adopt-link",
        help=h(_hh.on_conflict()),
    )
    pull.add_argument("--no-subscribe", action="store_true", help=h(_hh.no_subscribe()))
    pull.add_argument("--auto-push", action="store_true", help=h(_hh.auto_push()))
    pull.add_argument("--no-llm", action="store_true", help=h(_hh.no_llm()))
    pull.add_argument("--yes", action="store_true", help=h(_hh.yes()))


# -- shared resolution ---------------------------------------------------------------------


def resolve_tenant(config_dir: Path, declared: str | None) -> tuple[str | None, list[Any]]:
    """``(tenant, orgs)``: the declared one, the sole eligible one, or None (ambiguous)."""

    from local_operator.hub_org.eligibility import eligible_orgs

    orgs = eligible_orgs(config_dir)
    tenant = (declared or "").strip()
    if tenant:
        return tenant, orgs
    if len(orgs) == 1:
        return orgs[0].tenant_id, orgs
    return None, orgs


def _gate_or_refresh(config_dir: Path) -> bool:
    """Open gate, or one forced refresh; False after printing the one sentence."""

    from local_operator.hub_org.eligibility import is_eligible, refresh

    if is_eligible(config_dir):
        return True
    refresh(config_dir, boundary="explicit", force=True)
    if is_eligible(config_dir):
        return True
    print(_t(_ho.ineligible()), file=sys.stderr)
    return False


def _print_orgs(orgs: Sequence[Any]) -> None:
    print(_t(_ho.choose_org()), file=sys.stderr)
    for org in orgs:
        print(f"  {org.tenant_id}  {org.tenant_name}  ({org.role})", file=sys.stderr)


def resolve_org_ref(
    client: Any,
    tenant: str,
    kind: str,
    ref: str,
    *,
    config_dir: Path,
    quiet: bool = False,
) -> str | None:
    """The hub id ``ref`` names in ``tenant`` (id, exact name or unique prefix).

    WHY: the org routes take only ids, so `lop skills pull --org T
    minerva-platform-deployments` answered "no such skill". The resolution and
    its refusal (ambiguous prefixes list candidates, never guess) are one rule:
    :func:`hub_org.catalogue.resolve_reference`. Prints the chosen id on stderr,
    so a ``--json`` stdout payload stays parseable. Returns None after printing
    the refusal.
    """

    from local_operator.hub_org import catalogue as cat

    if cat.looks_like_hub_id(ref):
        return ref.strip()
    try:
        items = cat.list_catalogue(client, tenant, (kind,), config_dir=config_dir)
    except Exception:  # noqa: BLE001 - see below
        # BACKWARD COMPATIBLE: a list that cannot be read must not break the
        # id path that worked before names did. The per-id GET answers.
        return ref.strip()
    try:
        item = cat.resolve_reference(items, ref, kind=kind)  # type: ignore[arg-type]
    except cat.ReferenceError_ as error:
        if not error.candidates:
            # No name matched: treat it as an id (any shape the hub serves) and
            # let the per-id read give the authoritative not-found.
            return ref.strip()
        print(_t(_ho.ambiguous(ref=ref)), file=sys.stderr)
        for candidate in error.candidates:
            print(f"  {candidate.hub_id}  {candidate.name}", file=sys.stderr)
        return None
    if not quiet and item.hub_id != ref.strip():
        print(_t(_ho.resolved(ref=ref, hub_id=item.hub_id)), file=sys.stderr)
    return item.hub_id


# -- handlers --------------------------------------------------------------------------------


def _client(base_dir: Path) -> Any:
    # The CLI resolver prints the login/destination remedy; reuse it.
    from local_operator.cli import _resolve_org_client

    return _resolve_org_client(base_dir)


def _last_enrol(config_dir: Path) -> dict[str, Any]:
    try:
        # Slice B's module; absent until that PR lands, so an ImportError is "no record".
        from local_operator.hub_org.enrol import (  # type: ignore[import-not-found]
            last_enrol,
        )
    except ImportError:
        return {}
    try:
        value = last_enrol(config_dir)
    except Exception:  # noqa: BLE001 - a status read never fails the listing
        return {}
    return value if isinstance(value, dict) else {}


def unlinked_line(names: Sequence[str]) -> str:
    return _t(_ho.unlinked_copies(names=", ".join(names)))


def list_command(args: argparse.Namespace, base_dir: Path) -> int:
    from local_operator.config import ConfigManager
    from local_operator.hub_org import catalogue as cat

    config_dir = ConfigManager(base_dir).config_dir
    if not _gate_or_refresh(config_dir):
        return 1
    tenant, orgs = resolve_tenant(config_dir, getattr(args, "org", None))
    targets = [tenant] if tenant else [o.tenant_id for o in orgs]
    client = _client(base_dir)
    if client is None:
        return 1
    kinds = tuple(getattr(args, "kind", None) or cat.KINDS)
    payload: dict[str, Any] = {}
    for target in targets:
        result = cat.catalogue_result(client, target, kinds, config_dir=config_dir)
        payload[target] = {**result.to_json(), "last_enrol": _last_enrol(config_dir).get(target)}
    if getattr(args, "json", False):
        print(json.dumps(payload, indent=2))
        return 0
    names = {o.tenant_id: o.tenant_name for o in orgs}
    for target, body in payload.items():
        print(f"{names.get(target, target)} ({target})")
        for item in body["items"]:
            local = item["local"] + (f" as {item['local_name']}" if item["local_name"] else "")
            print(f"  {item['kind']:<5}  {item['name']:<36}  {local:<24}  {item['hub_id']}")
        for kind, cause in body["errors"].items():
            print("  " + _t(_ho.list_failed(kind=kind, cause=cause)))
        unlinked = [i["name"] for i in body["items"] if i["local"] == "conflict"]
        if unlinked:
            print("  " + unlinked_line(unlinked))
        enrol = body.get("last_enrol") or {}
        if enrol:
            pulled = enrol.get("pulled")
            count = len(pulled) if isinstance(pulled, list) else int(pulled or 0)
            print("  " + _t(_ho.last_enrol(at=str(enrol.get("at") or ""), pulled=count)))
            _print_adopted(enrol.get("adopted") or [], indent="    ")
    return 0


def _print_adopted(adopted: Sequence[Any], *, indent: str = "  ") -> None:
    rows = [a for a in adopted if isinstance(a, dict)]
    if not rows:
        return
    print(indent + _t(_ho.adopted_heading()))
    for row in rows:
        backup = f" ({row['backup']})" if row.get("backup") else ""
        print(f"{indent}  {row.get('kind')} {row.get('name')}: {row.get('result')}{backup}")


def pull_command(args: argparse.Namespace, base_dir: Path) -> int:
    from local_operator.config import ConfigManager
    from local_operator.hub_org import catalogue as cat
    from local_operator.hub_org.pull import pull_all

    manager = ConfigManager(base_dir)
    config_dir = manager.config_dir
    if not _gate_or_refresh(config_dir):
        return 1
    tenant, orgs = resolve_tenant(config_dir, getattr(args, "org", None))
    if tenant is None:
        _print_orgs(orgs)
        return 1
    client = _client(base_dir)
    if client is None:
        return 1
    kinds = tuple(getattr(args, "kind", None) or cat.KINDS)
    names = list(getattr(args, "name", None) or [])
    as_json = bool(getattr(args, "json", False))
    if names:
        # Names resolve through the same rule as every other org pull, so a
        # typo or an ambiguous prefix refuses before anything is written.
        items = cat.list_catalogue(client, tenant, kinds, config_dir=config_dir)
        resolved: list[str] = []
        for ref in names:
            try:
                item = cat.resolve_reference(items, ref)
            except cat.ReferenceError_ as error:
                if error.candidates:
                    print(_t(_ho.ambiguous(ref=ref)), file=sys.stderr)
                    for c in error.candidates:
                        print(f"  {c.hub_id}  {c.kind} {c.name}", file=sys.stderr)
                else:
                    print(_t(_ho.not_found(org=tenant, ref=ref)), file=sys.stderr)
                return 1
            if item.hub_id != ref:
                print(_t(_ho.resolved(ref=ref, hub_id=item.hub_id)), file=sys.stderr)
            resolved.append(item.hub_id)
        names = resolved
    if not getattr(args, "yes", False):
        # The plan: a read, no writes. Printing what WOULD happen first is the
        # same contract as every other preview here; --yes applies it.
        result = cat.catalogue_result(client, tenant, kinds, config_dir=config_dir)
        todo = [i for i in result.items if i.local != "linked" and (not names or i.hub_id in names)]
        conflicts = [i for i in todo if i.local == "conflict"]
        if as_json:
            print(
                json.dumps({"plan": [i.to_json() for i in todo], "errors": result.errors}, indent=2)
            )
            return 0
        for item in todo:
            print(f"  {item.kind:<5}  {item.name:<36}  {item.local}")
        print(_t(_ho.plan(count=len(todo), conflicts=len(conflicts))))
        return 0
    report = pull_all(
        manager,
        tenant,
        kinds=kinds,
        names=names or None,
        subscribe=not getattr(args, "no_subscribe", False),
        auto_push=bool(getattr(args, "auto_push", False)),
        on_conflict=getattr(args, "on_conflict", "adopt-link"),
        allow_llm=not getattr(args, "no_llm", False),
        client=client,
    )
    if as_json:
        print(json.dumps(report.to_json(), indent=2))
    else:
        for item in report.items:
            detail = f"  {item.detail}" if item.detail else ""
            print(f"  {item.kind:<5}  {item.name:<36}  {item.outcome}{detail}")
        for kind, cause in report.errors.items():
            print("  " + _t(_ho.list_failed(kind=kind, cause=cause)))
        _print_adopted(report.adopted)
    return 1 if any(i.outcome == "failed" for i in report.items) or report.errors else 0


def command(args: argparse.Namespace, base_dir: Path) -> int:
    if getattr(args, "hub_org_command", None) == "pull":
        return pull_command(args, base_dir)
    return list_command(args, base_dir)


# -- `lop hub status` / `teams search` hooks --------------------------------------------------


def status_unlinked_lines(base_dir: Path) -> list[str]:
    """The "unlinked local copies" line(s) for `lop hub status`, eligible accounts only.

    Network: one catalogue read per eligible org, and only for an eligible
    account, so the signed-out and no-team paths stay offline. Never raises:
    `hub status` is a status read and must not fail over an org list.
    """

    from local_operator.config import ConfigManager
    from local_operator.hub_org import catalogue as cat
    from local_operator.hub_org.client import build_org_client
    from local_operator.hub_org.eligibility import eligible_orgs

    manager = ConfigManager(base_dir)
    orgs = eligible_orgs(manager.config_dir)
    if not orgs:
        return []
    try:
        client = build_org_client(manager).client
        if client is None:
            return []
        names: list[str] = []
        for org in orgs:
            result = cat.catalogue_result(client, org.tenant_id, config_dir=manager.config_dir)
            names += [i.name for i in result.unlinked_local_copies()]
    except Exception:  # noqa: BLE001 - see docstring
        return []
    return [unlinked_line(names)] if names else []


def org_team_rows(base_dir: Path, query: str) -> list[dict[str, Any]]:
    """Org teams matching ``query`` for `teams search`, labelled by org; [] when ineligible."""

    from local_operator.config import ConfigManager
    from local_operator.hub_org.client import build_org_client
    from local_operator.hub_org.eligibility import eligible_orgs

    manager = ConfigManager(base_dir)
    orgs = eligible_orgs(manager.config_dir)
    if not orgs:
        return []
    try:
        client = build_org_client(manager).client
    except Exception:  # noqa: BLE001 - org rows are additive; public output still prints
        return []
    if client is None:
        return []
    needle = query.casefold()
    rows: list[dict[str, Any]] = []
    for org in orgs:
        try:
            teams = client.list_org_teams(org.tenant_id)
        except Exception:  # noqa: BLE001 - one org's failure never hides the public list
            continue
        for team in teams:
            text = f"{team.get('name', '')} {team.get('description', '')}".casefold()
            if needle and needle not in text:
                continue
            rows.append({**team, "scope": f"org:{org.tenant_name}", "tenant_id": org.tenant_id})
    return rows
