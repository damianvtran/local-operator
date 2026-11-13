"""Mirror the operator's failover routing (``retry.*``) onto paired mesh devices.

WHY THIS EXISTS. A session on a mesh node must take on work exactly like a local
one. Credentials already travel (the borrow/lend lane) and agents, teams, custom
instructions and MCP servers ride the definitions sync, but ``retry.fallbackChains``
reached no peer at all: a node's ``config.yml`` keeps the shipped ``{}``, so its
effective chain was primary-only and a borrowed Anthropic rate limit ENDED the turn
there, where the same turn on the operator's Mac would have walked
``anthropic → openai → deepseek → auto``. Same for the opt-ins that shape that walk
(``usageAwareFallback`` defaults ``False``, the reserve, the account policy).

WHAT TRAVELS — "units", each mirrored and owned independently:

* ``fallbackChains.<key>`` — one unit per chain key (``default``, ``anthropic/*``,
  an exact ``provider/model``). Per key, not the whole map, so a node can keep its
  own chain for one model while taking the operator's ``default``.
* ``modelFallback``, ``usageAwareFallback``, ``usageReservePercent``,
  ``pinnedFallback`` — the scalars the walk reads.
* ``accountSelectionPolicy`` — the sender's EFFECTIVE policy
  (``account_policy.effective_account_selection_policy``), never the legacy
  ``usageAwareAccountPick`` boolean: the reader resolves "presence before
  defaults", so a node whose generated config already spells the canonical key
  would silently ignore a mirrored legacy boolean.

DELIBERATELY NOT CARRIED: ``retry.enabled``/``maxRetries``/``baseDelayMs``/
``connectivity*`` (device-local transport tuning: a flaky cloud link wants a
different patience than a Mac on wired ethernet), ``subagents.models`` (its own
lane, PR #3869), and anything credential-shaped (a row whose text matches a
credential shape is withheld by the sender and refused by the receiver). Chain
entries are provider/model/effort selectors; they name WHAT to try, never HOW to
authenticate, so a node lacking a credential for a hop simply has the existing
walk skip it (``failover`` already does) — the hop is never dropped from config.

CONFLICT POLICY (the operator-instructions policy, applied per unit):

* the node has no value for the unit, or holds the SHIPPED DEFAULT → install. A
  value equal to the default carries no intent this code can tell apart from a
  generated config file, and treating it as "the node's own" would defeat the
  whole sync on every node whose ``config.yml`` was written with defaults;
* the node holds an untouched mirror from the SAME origin → update;
* anything else (the node's own value, an edited mirror, a mirror from another
  device) → kept, and recorded so the sender stops re-offering THIS revision and
  ``lop network definitions state``/readiness can say why the node routes
  differently;
* a unit the origin no longer sends → an untouched mirror of it is REMOVED (the
  operator deleted that chain on the head), an edited one is left and forgotten
  (it is the node's own now).

The direction is the definitions sync's: every device sends the units it AUTHORED
(never a mirror, so a node cannot hand the head's chain to a third device as its
own), to every paired member, on the syncer's link-up nudge, its cadence (a
config change moves the bundle digest) and after a relay restart (the link comes
up again). Writes go through the settings registry's own writer under the config
write lock, so a concurrent ``/settings`` save is never clobbered.

WIRE: additive. A receiver advertises support by carrying ``routing`` in its
``net_definitions`` state reply; a sender only sends the ``routing`` bundle key to
such a peer. An older peer neither advertises nor receives it, and an older sender
never sends it — no existing op changes shape.
"""

from __future__ import annotations

import hashlib
import logging
import time
from pathlib import Path
from typing import Any, Mapping

logger = logging.getLogger(__name__)

ROUTING_KIND = "routing"
#: definitions.json key holding the per-unit provenance and per-origin offers.
ROUTING_INDEX_KEY = "routing"
CHAIN_PREFIX = "fallbackChains."
#: camelCase key -> the legacy snake spelling the runtime still reads
#: (``RetrySettings.from_settings``). A node that wrote the snake spelling OWNS
#: that unit: the camel key the mirror would write outranks it in the reader.
SCALAR_KEYS: dict[str, str] = {
    "modelFallback": "model_fallback",
    "usageAwareFallback": "usage_aware_fallback",
    "usageReservePercent": "usage_reserve_percent",
    "pinnedFallback": "pinned_fallback",
    "accountSelectionPolicy": "account_selection_policy",
}
#: Bounds on what one origin may install on a node. A chain map past these is not
#: an operator's routing, it is a malformed or hostile producer.
MAX_CHAINS = 64
MAX_CHAIN_ENTRIES = 32
MAX_CHAIN_KEY_CHARS = 200
MAX_ROUTING_BYTES = 64 * 1024


class RoutingRefused(ValueError):
    """An incoming routing row this device will not install (reason in ``str``)."""


def _digest(value: Any) -> str:
    from local_operator.network.definitions import canonical

    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def _defaults() -> dict[str, Any]:
    """The shipped default per scalar unit, from the ``/settings`` registry.

    The registry is the one place defaults are pinned to their consumers
    (``test_every_default_matches_its_consumer``), so reading it here keeps "is
    this the default?" from becoming a second definition that can drift.
    """
    from local_operator import settings_io
    from local_operator.config import DEFAULT_ACCOUNT_SELECTION_POLICY

    out: dict[str, Any] = {"accountSelectionPolicy": DEFAULT_ACCOUNT_SELECTION_POLICY}
    for setting in settings_io.SETTINGS:
        if setting.path[:1] == ("retry",) and len(setting.path) == 2:
            key = setting.path[1]
            if key in SCALAR_KEYS and key not in out:
                out[key] = setting.default
    return {key: _normalise_scalar(key, value) for key, value in out.items()}


def _normalise_scalar(key: str, value: Any) -> Any:
    """A scalar in its comparable form, or ``None`` when it is not a usable value.

    One normaliser for both ends, so the digest a node records on install is the
    digest it re-reads from its own config (no re-send loop): ``10`` and ``10.0``
    are the same reserve.
    """
    if key in ("modelFallback", "usageAwareFallback"):
        return value if type(value) is bool else None
    if key == "usageReservePercent":
        if type(value) is bool or not isinstance(value, (int, float)):
            return None
        number = float(value)
        return number if 0.0 <= number <= 100.0 else None
    if key == "pinnedFallback":
        from local_operator.providers.failover import PINNED_FALLBACK_VALUES

        text = value.strip() if isinstance(value, str) else ""
        return text if text in PINNED_FALLBACK_VALUES else None
    if key == "accountSelectionPolicy":
        from local_operator.providers.account_policy import ACCOUNT_SELECTION_POLICIES

        return value if isinstance(value, str) and value in ACCOUNT_SELECTION_POLICIES else None
    return None


def _normalise_chain(key: Any, entries: Any) -> list[Any] | None:
    """A chain through the runtime's own normaliser, or ``None`` when unusable.

    ``failover._normalize_chains`` is what every model call applies, so a chain
    accepted here is one the node's walk resolves, and the stored form (selector
    strings, mappings only where an ``effort`` is load-bearing) round-trips to the
    same digest.
    """
    from local_operator.providers.failover import _normalize_chains

    if not isinstance(key, str) or not key.strip() or len(key) > MAX_CHAIN_KEY_CHARS:
        return None
    if isinstance(entries, (str, bytes)) or not isinstance(entries, (list, tuple)):
        return None
    if len(entries) > MAX_CHAIN_ENTRIES:
        return None
    normalised = _normalize_chains({key: list(entries)}).get(key)
    if not normalised or len(normalised) != len(entries):
        # A hop the runtime would drop is a chain this device would run
        # differently from its author; refuse rather than install a shorter walk.
        return None
    return list(normalised)


def _units_from_retry(retry: Mapping[str, Any]) -> dict[str, Any]:
    """Every unit EXPLICITLY present in a ``retry`` mapping, as written (normalised
    where possible, raw otherwise so a broken value still counts as "the node's own").
    """
    from local_operator.providers.account_policy import (
        effective_account_selection_policy,
    )

    units: dict[str, Any] = {}
    for key, alias in SCALAR_KEYS.items():
        if key == "accountSelectionPolicy":
            present = any(
                name in retry
                for name in (
                    "accountSelectionPolicy",
                    "account_selection_policy",
                    "usageAwareAccountPick",
                    "usage_aware_account_pick",
                )
            )
            if present:
                units[key] = effective_account_selection_policy({"retry": dict(retry)})
            continue
        if key in retry or alias in retry:
            raw = retry[key] if key in retry else retry[alias]
            normalised = _normalise_scalar(key, raw)
            units[key] = normalised if normalised is not None else repr(raw)[:200]
    chains = retry.get("fallbackChains", retry.get("fallback_chains"))
    if isinstance(chains, Mapping):
        for name, entries in chains.items():
            if not isinstance(name, str):
                continue
            normalised_chain = _normalise_chain(name, entries)
            units[CHAIN_PREFIX + name] = (
                normalised_chain if normalised_chain is not None else repr(entries)[:2000]
            )
    return units


def _disk_units(root: Path) -> dict[str, Any] | None:
    """The units ``root``'s ``config.yml`` spells explicitly; ``None`` if unreadable.

    ``read_config_values`` and never ``ConfigManager``: this runs on a path a PEER
    reaches (state, apply), and constructing a manager on a malformed file renames
    it aside to ``config.yml.bad.<stamp>`` — a peer's probe must never do that.
    Raw values, not defaults-merged ones, because "the node set this" is the
    question.
    """
    from local_operator.config import read_config_values

    values = read_config_values(root)
    if values is None:
        return None
    retry = values.get("retry")
    return _units_from_retry(retry) if isinstance(retry, Mapping) else {}


def _is_default(unit: str, value: Any, defaults: Mapping[str, Any]) -> bool:
    if unit.startswith(CHAIN_PREFIX):
        # The shipped chain map is ``{}``: any chain at all is a choice.
        return False
    return unit in defaults and value == defaults[unit]


def _recorded(index: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    block = index.get(ROUTING_INDEX_KEY)
    block = block if isinstance(block, Mapping) else {}
    units = block.get("units")
    offered = block.get("offered")
    return (
        (
            {str(k): dict(v) for k, v in units.items() if isinstance(v, Mapping)}
            if isinstance(units, Mapping)
            else {}
        ),
        (
            {str(k): dict(v) for k, v in offered.items() if isinstance(v, Mapping)}
            if isinstance(offered, Mapping)
            else {}
        ),
    )


def _classify(
    disk: Mapping[str, Any], recorded: Mapping[str, Any], defaults: Mapping[str, Any]
) -> dict[str, dict[str, str]]:
    """``unit -> {source, origin}`` for every unit present on disk or recorded."""
    out: dict[str, dict[str, str]] = {}
    for unit, value in disk.items():
        rec = recorded.get(unit)
        if rec is not None and str(rec.get("digest") or "") == _digest(value):
            out[unit] = {"source": "mirror", "origin": str(rec.get("origin") or "")}
        elif _is_default(unit, value, defaults):
            out[unit] = {"source": "default", "origin": ""}
        else:
            out[unit] = {
                "source": "edited" if rec is not None else "local",
                "origin": str(rec.get("origin") or "") if rec is not None else "",
            }
    return out


def routing_state(root: Path) -> dict[str, Any]:
    """What this device holds, for the peer's ``state`` reply and the CLI.

    ``offered`` maps an origin device to the digest of the last row it sent that
    this device fully settled — installed, or kept its own for units it STILL
    owns. A unit it has since given up (cleared, reset to default) withdraws the
    marker, so the origin's next tick re-offers and the mirror lands: without
    that, clearing a local override would never let the operator's value in.
    """
    from local_operator.network.definitions import read_index

    disk = _disk_units(root)
    recorded, offered = _recorded(read_index(root))
    if disk is None:
        return {"supported": 1, "status": "unreadable", "units": {}, "offered": {}}
    classes = _classify(disk, recorded, _defaults())
    settled: dict[str, str] = {}
    for origin, offer in offered.items():
        kept = offer.get("kept")
        kept_units = [str(u) for u in kept] if isinstance(kept, list) else []
        if all(classes.get(u, {}).get("source") in ("local", "edited") for u in kept_units):
            settled[origin] = str(offer.get("digest") or "")
    return {"supported": 1, "status": "ok", "units": classes, "offered": settled}


def local_row(root: Path) -> dict[str, Any] | None:
    """The units THIS device authored, as the row it sends; ``None`` when unreadable.

    A mirror is never re-sent (see the module docstring). An EMPTY row is still a
    row: it is how a head that deleted its last chain tells a node to drop the
    mirror of it.
    """
    from local_operator.network.definitions import read_index

    disk = _disk_units(root)
    if disk is None:
        return None
    recorded, _ = _recorded(read_index(root))
    classes = _classify(disk, recorded, _defaults())
    units = {
        unit: disk[unit]
        for unit, cls in classes.items()
        if cls["source"] in ("local", "edited") and _valid_unit(unit, disk[unit])
    }
    return {"kind": ROUTING_KIND, "name": ROUTING_KIND, "units": dict(sorted(units.items()))}


def _valid_unit(unit: str, value: Any) -> bool:
    if unit.startswith(CHAIN_PREFIX):
        return _normalise_chain(unit[len(CHAIN_PREFIX) :], value) == value
    return unit in SCALAR_KEYS and _normalise_scalar(unit, value) == value


def withheld_shape(row: Mapping[str, Any]) -> str:
    """The credential-shape label that keeps ``row`` off the wire, or ``""``.

    Model ids never look like a key; a pasted token in a chain entry would, and a
    routing mirror must never become the way a secret reaches another device.
    """
    from local_operator.network.definitions import _flatten_text, credential_shape

    units = row.get("units")
    texts = _flatten_text(dict(units)) if isinstance(units, Mapping) else []
    texts += [str(key) for key in (units or {})]
    for text in texts:
        label = credential_shape(text)
        if label:
            return label
    return ""


def to_send(
    row: Mapping[str, Any] | None, remote: Mapping[str, Any], *, origin: str
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """``(row to send or None, report)`` for one peer, from its ``state`` reply."""
    remote_in = remote.get(ROUTING_KIND)
    if not isinstance(remote_in, Mapping):
        return None, {"outcome": "unsupported"}
    if row is None:
        return None, {"outcome": "none", "reason": "this device's config could not be read"}
    label = withheld_shape(row)
    if label:
        return None, {
            "outcome": "withheld",
            "reason": f"it is shaped like a credential ({label}), so it was not sent",
        }
    offered = remote_in.get("offered")
    if isinstance(offered, Mapping) and str(offered.get(origin) or "") == _digest(row):
        return None, {"outcome": "in_sync"}
    if not row.get("units") and not _holds_mirror_from(remote_in, origin):
        # Nothing authored here and nothing of ours to withdraw there: an empty
        # row would only cost the peer a write and every other sync stream an
        # apply frame it does not expect (a device with no routing of its own is
        # the common case, not an event).
        return None, {"outcome": "none"}
    return dict(row), {"outcome": "sent"}


def _holds_mirror_from(remote_in: Mapping[str, Any], origin: str) -> bool:
    units = remote_in.get("units")
    if not isinstance(units, Mapping):
        return False
    return any(
        isinstance(cls, Mapping)
        and cls.get("source") == "mirror"
        and str(cls.get("origin") or "") == origin
        for cls in units.values()
    )


def _validate_incoming(row: Any) -> dict[str, Any]:
    if not isinstance(row, Mapping):
        raise RoutingRefused("the routing row is not a mapping")
    units = row.get("units")
    if not isinstance(units, Mapping):
        raise RoutingRefused("the routing row carries no units")
    from local_operator.network.definitions import canonical

    if len(canonical(dict(units)).encode("utf-8")) > MAX_ROUTING_BYTES:
        raise RoutingRefused(f"the routing row is over {MAX_ROUTING_BYTES} bytes")
    if sum(1 for unit in units if str(unit).startswith(CHAIN_PREFIX)) > MAX_CHAINS:
        raise RoutingRefused(f"more than {MAX_CHAINS} fallback chains")
    out: dict[str, Any] = {}
    for unit, value in units.items():
        unit = str(unit)
        if unit.startswith(CHAIN_PREFIX):
            chain = _normalise_chain(unit[len(CHAIN_PREFIX) :], value)
            if chain is None:
                raise RoutingRefused(f"retry.{unit} is not a usable fallback chain")
            out[unit] = chain
        elif unit in SCALAR_KEYS:
            scalar = _normalise_scalar(unit, value)
            if scalar is None:
                raise RoutingRefused(f"retry.{unit} has an unusable value")
            out[unit] = scalar
        else:
            raise RoutingRefused(f"retry.{unit} is not a routing setting this device mirrors")
    label = withheld_shape({"units": out})
    if label:
        raise RoutingRefused(f"it is shaped like a credential ({label}), which is never installed")
    return out


def apply_routing_locked(
    root: Path, raw_row: Any, origin: str, index: dict[str, Any]
) -> dict[str, Any]:
    """Install ``origin``'s routing units here per the module's conflict policy.

    The caller holds this root's definition lock and writes ``index`` back when
    the result carries ``index_dirty``. The config write itself happens under the
    config write lock, with the on-disk read repeated INSIDE it, so the decision
    and the write see the same file (a ``/settings`` save cannot land between).
    """
    from local_operator import settings_io
    from local_operator.config import CONFIG_FILE_NAME, ConfigManager
    from local_operator.config_writer import config_write_lock

    try:
        incoming = _validate_incoming(raw_row)
    except RoutingRefused as exc:
        return {"outcome": "refused", "reason": str(exc)}
    if not origin:
        return {"outcome": "refused", "reason": "the sending device is not known"}
    row_digest = _digest({"kind": ROUTING_KIND, "name": ROUTING_KIND, "units": incoming})
    # Same guard as the tier lane: constructing a manager on a malformed file
    # would rename it aside before the writer could refuse.
    try:
        settings_io.require_readable_config_path(root / CONFIG_FILE_NAME)
    except settings_io.ConfigUnreadableError:
        return {"outcome": "refused", "reason": "this device's config.yml could not be read"}
    defaults = _defaults()
    recorded, offered = _recorded(index)
    installed: list[str] = []
    updated: list[str] = []
    removed: list[str] = []
    kept: list[dict[str, str]] = []
    manager = ConfigManager(root)
    with config_write_lock(manager.config_dir):
        disk = _disk_units(root)
        if disk is None:
            return {"outcome": "refused", "reason": "this device's config.yml could not be read"}
        settings_io._reload_before_write(manager)  # noqa: SLF001 — the registry's writer
        existing = manager.get_config_value("retry", None)
        section: dict[str, Any] = dict(existing) if isinstance(existing, Mapping) else {}
        chains_in = section.get("fallbackChains")
        chains: dict[str, Any] = dict(chains_in) if isinstance(chains_in, Mapping) else {}
        drop: dict[tuple[str, ...], str] = {}
        reason = f"mirrored failover routing from {origin}"
        new_records = dict(recorded)

        def _set(unit: str, value: Any) -> None:
            if unit.startswith(CHAIN_PREFIX):
                chains[unit[len(CHAIN_PREFIX) :]] = value
            else:
                section[unit] = value

        def _unset(unit: str) -> None:
            if unit.startswith(CHAIN_PREFIX):
                name = unit[len(CHAIN_PREFIX) :]
                chains.pop(name, None)
                drop[("retry", "fallbackChains", name)] = reason
            else:
                section.pop(unit, None)
                drop[("retry", unit)] = reason

        for unit, value in incoming.items():
            current = disk.get(unit)
            rec = recorded.get(unit)
            rec_origin = str(rec.get("origin") or "") if rec is not None else ""
            untouched_mirror = (
                rec is not None
                and current is not None
                and (str(rec.get("digest") or "") == _digest(current))
            )
            if current is None or _is_default(unit, current, defaults):
                _set(unit, value)
                installed.append(unit)
            elif untouched_mirror and rec_origin == origin:
                if current != value:
                    _set(unit, value)
                    updated.append(unit)
            elif untouched_mirror:
                kept.append({"unit": unit, "reason": f"it holds a mirror from {rec_origin}"})
                continue
            else:
                kept.append({"unit": unit, "reason": "this device set its own value"})
                continue
            new_records[unit] = {"origin": origin, "digest": _digest(value)}
        for unit, rec in recorded.items():
            if str(rec.get("origin") or "") != origin or unit in incoming:
                continue
            current = disk.get(unit)
            if current is not None and str(rec.get("digest") or "") == _digest(current):
                # The operator deleted this unit on the origin: follow it.
                _unset(unit)
                removed.append(unit)
            # Absent already, or edited here (now the node's own): forget either way.
            new_records.pop(unit, None)
        if installed or updated or removed:
            if chains:
                section["fallbackChains"] = chains
            else:
                section.pop("fallbackChains", None)
            values = manager.get_config().values
            for key in SCALAR_KEYS:
                setting = next((s for s in settings_io.SETTINGS if s.key == f"retry.{key}"), None)
                if setting is not None and key in section:
                    problem = settings_io.validate(setting, section[key], values)
                    if problem is not None:
                        return {"outcome": "refused", "reason": f"retry.{key}: {problem}"}
            settings_io._store(manager, ("retry",), section, drop=drop)  # noqa: SLF001
    if installed or updated or removed:
        settings_io._invalidate_caches()  # noqa: SLF001 — same post-write step as write_setting
    offered[origin] = {
        "digest": row_digest,
        "kept": [entry["unit"] for entry in kept],
        "at": time.time(),
    }
    index[ROUTING_INDEX_KEY] = {"units": new_records, "offered": offered}
    if installed and not recorded:
        outcome = "installed"
    elif installed or updated or removed:
        outcome = "updated"
    elif kept:
        outcome = "kept"
    else:
        outcome = "unchanged"
    return {
        "outcome": outcome,
        "installed": installed,
        "updated": updated,
        "removed": removed,
        "kept": kept,
        "index_dirty": True,
    }


def failover_fact(root: Path, *, hosting: str = "", model_name: str = "") -> dict[str, Any] | None:
    """The chain this device's default model would actually walk, with provenance.

    Resolved through the runtime's own ``RetrySettings.from_settings`` and
    ``resolve_chain_key``, so readiness shows the chain a turn takes, not a second
    reading of the config. Selectors only — no credential state; whether a hop is
    servable here is the model-credential row's question. ``None`` when the config
    cannot be read (a readiness answer never fails on this).
    """
    from local_operator.config import read_config_values
    from local_operator.network.definitions import read_index
    from local_operator.providers.failover import RetrySettings, resolve_chain_key

    try:
        values = read_config_values(root)
        if values is None:
            return None
        retry = RetrySettings.from_settings(values)
        selector = f"{hosting}/{model_name}" if hosting and model_name else ""
        key = resolve_chain_key(selector, retry.fallback_chains) if selector else None
        if key is None and "default" in retry.fallback_chains:
            key = "default"
        entries = list(retry.fallback_chains.get(key, ())) if key else []
        hops = [
            entry if isinstance(entry, str) else f"{entry.get('provider')}/{entry.get('model')}"
            for entry in entries[:MAX_CHAIN_ENTRIES]
        ]
        disk = _disk_units(root) or {}
        recorded, _ = _recorded(read_index(root))
        cls = _classify(disk, recorded, _defaults()).get(CHAIN_PREFIX + key, {}) if key else {}
        return {
            "chain_key": key or "",
            "hops": hops,
            "source": cls.get("source", ""),
            "origin": cls.get("origin", ""),
            "model_fallback": retry.model_fallback,
            "usage_aware_fallback": retry.usage_aware_fallback,
            "account_selection_policy": retry.account_selection_policy,
        }
    except Exception:  # noqa: BLE001 — a readiness answer must not fail on this
        logger.debug("routing: could not compute the failover fact", exc_info=True)
        return None


def failover_sentence(fact: Mapping[str, Any], *, origin_label: str = "") -> str:
    """One readiness sentence naming the effective chain and whose it is."""
    hops = [str(hop) for hop in fact.get("hops") or []]
    if not fact.get("model_fallback", True):
        return "Model fallback is off here: a failed default ends the turn."
    if not hops:
        return "No fallback chain here: a rate-limited default ends the turn."
    source = str(fact.get("source") or "")
    whose = (
        f" (mirrored from {origin_label or fact.get('origin') or 'a peer'})"
        if source == "mirror"
        else " (this device's own)" if source in ("local", "edited") else ""
    )
    return "Fails over to: " + " → ".join(hops) + whose + "."
