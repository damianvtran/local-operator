"""Agent-performable org-hub pulls: the host-built plan, the linked-update arm, the ask copy.

WHY THIS EXISTS. ``agent op=org_pull`` used to be an all-or-nothing account
write behind a card. A turn woken by a Team Bridge message, a peer or a wake has
no card surface, so the pull was refused and the agent told a non-technical user
to run ``lop agents sync`` (which does not even touch skills). Items the user had
already pulled were reported ``already-linked`` and never updated.

This module is the host side of the fix:

* :func:`build_plan` computes WHAT a pull would do from the catalogue and the
  sync check, in host code. The model supplies none of it; the consent seam
  renders it on the ask/card and binds a grant to its tenant and kinds.
* :func:`apply_linked` applies the available updates of already-linked items
  through the ordinary hub-sync apply (``all_available=True``, which ignores any
  prefer/replace, and ``allow_llm=False``), so a conflict is REPORTED for review,
  never forced, exactly like the unattended runner. It also re-subscribes links
  the user had unsubscribed, via the same policy writer ``pull_all`` uses.
* :func:`ask_question` is the host-authored consent ask (localized copy).

CONSTRAINTS. Nothing here deletes, replaces, unlinks or pushes; adopt/reconcile
stay behind ``confirm_replace`` in the tool. Network reads only through the
clients the callers already built.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from local_operator.hub_org.catalogue import CatalogueItem

#: Names shown per section of the ask preview before "and N more". The ask card
#: is a decision surface, not a report: eight names is enough to recognise what
#: is changing (Ben's agent wanted to SEE the changes) without pushing the
#: options off a phone screen.
PREVIEW_NAMES = 8

#: Classification states where the user's own copy has edits the hub does not
#: have. The apply keeps or merges them (needs-review on conflict), so the
#: preview marks them instead of implying an overwrite.
_LOCAL_EDIT_STATES = ("both-changed", "local-only", "baseline-unknown")

#: The answer option id the grant compares against is the LABEL that was
#: persisted with the ask (``consent["affirm"]``), never a re-rendered string,
#: so a language change between asking and answering cannot void a yes.
QUESTION_ID = "hub_pull"


def _short(revision: Any) -> str:
    text = str(revision or "").strip().strip('"').strip()
    return text[:16] if text else "?"


@dataclass
class PullPlan:
    """What an org pull would do. ``to_dict`` is what rides on the intent."""

    tenant: str
    org_name: str
    kinds: tuple[str, ...]
    new: list[dict[str, Any]] = field(default_factory=list)
    updates: list[dict[str, Any]] = field(default_factory=list)
    resubscribe: list[dict[str, Any]] = field(default_factory=list)
    #: Linked items whose check said up-to-date (org_list's ``current``).
    current: list[dict[str, Any]] = field(default_factory=list)
    errors: dict[str, str] = field(default_factory=dict)
    #: Host-only: the checks the update arm applies, and the linked catalogue
    #: rows the re-subscribe arm writes. Never serialized onto the intent.
    checks: list[Any] = field(default_factory=list, repr=False)
    linked_items: dict[tuple[str, str], CatalogueItem] = field(default_factory=dict, repr=False)

    def is_empty(self) -> bool:
        return not (self.new or self.updates or self.resubscribe)

    def to_dict(self) -> dict[str, Any]:
        return {
            "tenant": self.tenant,
            "org_name": self.org_name,
            "kinds": list(self.kinds),
            "new": [dict(i) for i in self.new],
            "updates": [dict(i) for i in self.updates],
            "resubscribe": [dict(i) for i in self.resubscribe],
        }


def build_plan(
    config_manager: Any,
    tenant: str,
    kinds: Sequence[str],
    *,
    client: Any,
    agent_registry: Any = None,
    hub_ids: Sequence[str] | None = None,
    on_conflict: str = "adopt-link",
    org_name: str = "",
) -> PullPlan:
    """Compute the plan from the catalogue (one list per kind) and a sync check.

    The check is a NETWORK read of each linked item (the same one ``lop agents
    sync`` and the runner do) with ``record=False``: planning before consent
    writes nothing, not even the status store.
    """

    from pathlib import Path

    from local_operator.hub_org import catalogue as cat
    from local_operator.hub_sync import service as svc

    config_dir = Path(config_manager.config_dir)
    result = cat.catalogue_result(
        client, tenant, kinds, config_dir=config_dir, agent_registry=agent_registry
    )
    plan = PullPlan(
        tenant=tenant, org_name=org_name or tenant, kinds=tuple(kinds), errors=dict(result.errors)
    )
    wanted = set(hub_ids) if hub_ids else None
    linked: list[CatalogueItem] = []
    for item in result.items:
        if wanted is not None and item.hub_id not in wanted:
            continue
        if item.local == "absent" or (item.local == "conflict" and on_conflict == "adopt-link"):
            plan.new.append(
                {
                    "kind": item.kind,
                    "name": item.name,
                    # The id the grant binds to and the tool pins the pull to
                    # (:func:`consent_items`): a name can be re-used by another item.
                    "hub_id": item.hub_id,
                    "local_edits": item.local == "conflict",
                }
            )
        elif item.local == "linked":
            linked.append(item)
    if not linked:
        return plan
    plan.linked_items = {(i.kind, i.hub_id): i for i in linked}
    try:
        ctx = svc.sync_context(config_manager)
        if agent_registry is not None:
            ctx.agent_registry = agent_registry
        checks = svc.check_items(
            ctx,
            kinds=tuple(kinds),
            only_ids={i.local_id for i in linked if i.local_id},
            record=False,
        )
        links = svc.links_snapshot(ctx, kinds=tuple(kinds)).get("links") or []
    except Exception as error:  # noqa: BLE001 - new items can still be pulled
        plan.errors["linked"] = str(error)[:300]
        return plan
    from_rev = {(str(r.get("kind")), str(r.get("hub_id"))): r.get("remote_revision") for r in links}
    for c in checks:
        item = plan.linked_items.get((c.kind, c.hub_id))
        if item is None:
            continue  # another org's link that shares a local id: never touched
        row = {"kind": c.kind, "name": item.local_name or c.name or item.name}
        if c.verdict == "available" and not (c.kind == "skill" and c.content_state == "absent"):
            state = getattr(c.classification, "state", "") or ""
            plan.checks.append(c)
            plan.updates.append(
                {
                    **row,
                    "hub_id": c.hub_id,
                    "from": _short(from_rev.get((c.kind, c.hub_id))),
                    "to": _short(c.remote_revision or item.revision),
                    # The FULL target revision the grant binds to: a yes to "update
                    # to this content" must not carry a later, unseen revision.
                    "rev": str(c.remote_revision or item.revision or ""),
                    "local_edits": state in _LOCAL_EDIT_STATES,
                }
            )
        elif c.verdict == "up-to-date":
            plan.current.append(row)
        else:
            plan.errors[f"{c.kind} {row['name']}"] = c.reason or c.verdict
    for link in links:
        key = (str(link.get("kind")), str(link.get("hub_id")))
        policy = link.get("policy") or {}
        if key in plan.linked_items and policy.get("subscribe") is False:
            item = plan.linked_items[key]
            plan.resubscribe.append(
                {"kind": key[0], "name": item.local_name or item.name, "hub_id": key[1]}
            )
    return plan


def update_states(plan: PullPlan) -> dict[tuple[str, str], str]:
    """``(kind, name) -> current|update-available`` for org_list's linked rows."""

    out = {(r["kind"], r["name"]): "current" for r in plan.current}
    out.update({(r["kind"], r["name"]): "update-available" for r in plan.updates})
    return out


def apply_linked(
    config_manager: Any,
    plan: PullPlan,
    *,
    client: Any,
    agent_registry: Any = None,
    role: str = "member",
) -> list[str]:
    """Apply the plan's linked updates and re-subscriptions; one report line each.

    Model-facing lines (English by design, AGENTS.md i18n). ``needs-review``
    names the conflict and states it was NOT forced, so the agent reports it
    instead of retrying with a stronger lever.
    """

    from local_operator.hub_sync import service as svc

    lines: list[str] = []
    if plan.checks:
        ctx = svc.sync_context(config_manager)
        if agent_registry is not None:
            ctx.agent_registry = agent_registry
        report = svc.apply_items(ctx, all_available=True, allow_llm=False, checks=plan.checks)
        revisions = {(c.kind, c.hub_id): _short(c.remote_revision) for c in plan.checks}
        for r in report.reports:
            line = f"{r.kind} {r.name}: "
            if r.applied:
                line += f"updated to revision {revisions.get((r.kind, r.hub_id), '?')}"
            elif r.outcome == "needs-review":
                line += "needs-review (local edits conflict with the hub; kept, not forced)"
            else:
                line += r.outcome
            cause = r.message or r.skipped_reason or r.error_class
            if cause and not r.applied:
                line += f" ({cause})"
            lines.append(line)
    if plan.resubscribe:
        from local_operator.hub_org import pull as pull_mod

        run = pull_mod._Run(
            config_manager,
            plan.tenant,
            client=client,
            agent_registry=agent_registry,
            team_registry=None,
            allow_llm=False,
        )
        for row in plan.resubscribe:
            item = plan.linked_items.get((row["kind"], row.get("hub_id", "")))
            if item is None:
                continue
            try:
                note = pull_mod._apply_policy(run, item, subscribe=True, auto_push=False, role=role)
            except Exception as error:  # noqa: BLE001 - one link never stops the rest
                note = f"not re-subscribed: {error}"
            lines.append(f"{item.kind} {row['name']}: " + (note or "re-subscribed"))
    return lines


# -- the host-authored consent ask ---------------------------------------------------


def consent_items(plan: dict[str, Any]) -> list[str]:
    """The exact writes a plan performs, as the keys a consent grant binds to.

    WHY (PR #3693 review R2). A queued ask is answered LATER, and the pull that
    spends the yes re-plans from the catalogue at that moment. Binding the grant
    only to tenant and kinds let a yes to "get skill A" carry whatever the hub
    published in between. A grant now covers a plan only when every key of the
    new plan was in the plan the person saw; anything else is re-asked.

    New items bind by hub id (the preview names the item, not a revision, and a
    first install has nothing local to protect); updates bind by hub id AND the
    full target revision, because the person agreed to a specific update;
    re-subscriptions bind by hub id. A fewer-items plan is covered (an item
    pulled meanwhile simply drops out), never a different or larger one.
    """

    keys: set[str] = set()
    for row in plan.get("new") or ():
        keys.add(f"new:{row.get('kind')}:{row.get('hub_id') or row.get('name')}")
    for row in plan.get("updates") or ():
        ident = row.get("hub_id") or row.get("name")
        keys.add(f"update:{row.get('kind')}:{ident}@{row.get('rev') or row.get('to')}")
    for row in plan.get("resubscribe") or ():
        keys.add(f"resubscribe:{row.get('kind')}:{row.get('hub_id') or row.get('name')}")
    return sorted(keys)


def _names(rows: Sequence[dict[str, Any]]) -> str:
    from local_operator.i18n.keys import wire_hub_pull as k
    from local_operator.i18n.text import tr

    _t = tr("wire.hub_pull")
    shown: list[str] = []
    for row in rows[:PREVIEW_NAMES]:
        # No revision hashes (design review D4): a non-technical reader cannot
        # act on them and they doubled every line. The grant still binds to the
        # full revision (:func:`consent_items`) and the tool's own report names
        # "updated to revision …".
        label = f"{row['kind']} {row['name']}"
        if row.get("local_edits"):
            label += " *"
        shown.append(label)
    text = ", ".join(shown)
    extra = len(rows) - PREVIEW_NAMES
    if extra > 0:
        text = _t(k.ask_more(names=text, count=str(extra)))
    return text


def plan_lines(plan: dict[str, Any]) -> list[str]:
    """The localized preview: one line per non-empty section, then the edits note."""

    from local_operator.i18n.keys import wire_hub_pull as k
    from local_operator.i18n.text import tr

    _t = tr("wire.hub_pull")
    new = list(plan.get("new") or ())
    updates = list(plan.get("updates") or ())
    resub = list(plan.get("resubscribe") or ())
    lines = [_t(k.ask_title(org=str(plan.get("org_name") or plan.get("tenant") or "")))]
    if new:
        lines.append(_t(k.ask_new(count=len(new), names=_names(new))))
    if updates:
        lines.append(_t(k.ask_updates(count=len(updates), names=_names(updates))))
    if resub:
        lines.append(_t(k.ask_resubscribe(count=len(resub), names=_names(resub))))
    if any(r.get("local_edits") for r in (*new, *updates)):
        lines.append(_t(k.ask_local_edits()))
    return lines


def ask_question(plan: dict[str, Any]) -> tuple[dict[str, Any], str]:
    """``(question, affirm_label)``: the one-question ask the host files."""

    from local_operator.i18n.keys import wire_hub_pull as k
    from local_operator.i18n.text import tr

    _t = tr("wire.hub_pull")
    affirm = _t(k.ask_option_update())
    question = {
        "id": QUESTION_ID,
        "question": "\n".join(plan_lines(plan)),
        "options": [
            {"label": affirm, "description": _t(k.ask_option_update_description())},
            {
                "label": _t(k.ask_option_not_now()),
                "description": _t(k.ask_option_not_now_description()),
            },
        ],
        "recommended": 0,
        "multi": False,
        "secret": False,
    }
    return question, affirm
