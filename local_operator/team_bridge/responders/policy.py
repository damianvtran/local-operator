"""The owner's local alert-responder policy: ``<config_dir>/team_bridge/responders.yaml``.

WHY A LOCAL FILE (RFC §7.1, S-34/S-40). Authority to start work from an alert must
live with the human owner on their own device — never on the hub, never in alert
text, never in anything the agent can write. The hub only learns the non-sensitive
projection (enable bit, tier, pause, consent hash, :func:`policy_sha256`); repos,
credential profile, link hosts and budgets never leave this machine.

THE FENCE. The file AND its ``team_bridge`` directory itself are in the guardrail
EVIDENCE class (``guardrails.classify._in_policy_region``): an agent's
``write``/``edit``, any bash redirect, ``mv``, ``cp``/``cp -R``, ``rsync``, ``ln``,
``tar -C``/``unzip -d``, ``git -C … checkout`` or ``rm`` onto either — case-folded and
after ``..`` — is CRITICAL and refused, the same tier as the guardrail's own records;
so is the policy path handed to an unknown program (``python3 -c``). The owner edits
it with their own editor (or, later, the desktop Responder dialog, RFC Phase 4).

A bulk copy, sync, extract or checkout whose DESTINATION is the state root (or an
ancestor that would land ``team_bridge`` in it) — ``rsync -a ~/forged/ ~/.local-operator/``,
``ditto``, ``cp -R x/``, ``cp --parents``, ``tar -x``/``unzip`` there, ``git checkout``
in it — is CRITICAL too (``guardrails.classify._bulk_reason``, review S-12), and so
is ``cp -l``/``--link``/``-al`` with the policy as its source (S-11).

What a command classifier CANNOT see: a write whose path is computed at run time
inside an interpreter or a script file (``python3 x.py`` that builds the path).
The real boundary for those is the S-41 spawner's OS-level write-deny on the state
root; until it lands, :func:`load_policy` closes what it can on the descriptor it
reads (:func:`_read_fenced`): a symlinked file, ``team_bridge`` or state directory,
a non-regular file, a file with a second name (hardlink, ``st_nlink != 1``), and a
file or either directory not owned by this user or group/world-writable — each is
"no policy" (OFF).

FAIL CLOSED, DEFAULT OFF. A missing file, an unreadable one, a YAML error, an
unknown key, a wrong type, an unknown tier or severity — every one of them yields
"no channel is enabled". There is no partial parse: one bad channel entry voids the
whole file, because a typo that silently drops a ``min_severity`` line would widen
what an alert may wake. The file is re-read on EVERY alert (no cache), so a local
pause or disable takes effect on the next wake.

Shape (``version: 1``)::

    version: 1
    responders:
      bcv_0123456789abcdef:              # conversation id, never a mutable slug
        enabled: true
        consent_version: responder_consent_v1
        consent_sha256: <64 hex>         # the served consent text the owner accepted
        autonomy_tier: T0                # T0-passive (Watch) | T0 (Triage) | T1 (Prepare & ask)
        min_severity: sev2               # sev1 | sev2 | sev3 | sev4
        allowed_app_ids: [iap_...]       # optional; empty = any app installed in the channel
        allowed_hook_ids: [ihk_...]      # optional
        paused_until: 2026-11-21T18:00:00Z   # optional
        max_concurrent_claims: 1
        credential_profile: radient-responder-readonly
        allowed_repos: [radient-ml/agent-server]
        allowed_runbooks: [radient-infra/docs/runbook.md]
        allowed_link_hosts: [app.datadoghq.com]
        budgets:
          max_alerts_per_hour: 6
          max_tokens_per_alert: 200000
          max_tokens_per_day: 2000000

``schedule`` (RFC: ``always | online_only | cron``) is accepted only as ``always`` in
this slice: a schedule this build cannot evaluate must refuse, never be ignored.
"""

from __future__ import annotations

import datetime as _dt
import errno
import hashlib
import os
import re
import stat
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

__all__ = [
    "AUTONOMY_TIERS",
    "CLAIMING_TIERS",
    "CONSENT_VERSION",
    "ChannelPolicy",
    "PolicyError",
    "ResponderPolicy",
    "Admission",
    "admit_alert",
    "load_policy",
    "parse_policy",
    "policy_path",
    "policy_sha256",
    "severity_rank",
]

#: The served consent version the hub accepts (agent-server ``ResponderConsentVersionV1``).
CONSENT_VERSION = "responder_consent_v1"

#: agent-server ``models.ResponderAutonomyTier`` (closed): Watch, Triage, Prepare & ask.
AUTONOMY_TIERS = ("T0-passive", "T0", "T1")
#: ``ResponderAutonomyTier.MayClaim``: Watch never claims or posts publicly.
CLAIMING_TIERS = frozenset({"T0", "T1"})

#: agent-server ``models.IntegrationSeverityRank``: sev1 is the most severe. ``unknown``
#: (or anything else) never wakes on the hub, and never admits here either.
_SEVERITY_RANK = {"sev1": 1, "sev2": 2, "sev3": 3, "sev4": 4}

_CONVERSATION_ID = re.compile(r"^bcv_[0-9a-z]{8,64}$")
_APP_ID = re.compile(r"^iap_[0-9a-f]{8,64}$")
_HOOK_ID = re.compile(r"^ihk_[0-9a-f]{8,64}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
# A segment of only dots (``..``) is a path climb, never a GitHub name (review n3).
_REPO = re.compile(r"^(?!\.{1,2}/)[A-Za-z0-9_.-]+/(?!\.{1,2}$)[A-Za-z0-9_.-]+$")
_HOST = re.compile(
    r"^(?=.{1,253}$)[a-z0-9]([a-z0-9-]{0,61}[a-z0-9])?(\.[a-z0-9]([a-z0-9-]{0,61}[a-z0-9])?)+$"
)
_PROFILE = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")

#: Bounds that keep a typo from authorizing a storm. Upper limits sit well above the
#: RFC's example policy (6 alerts/h, 200k tokens/alert, 2M tokens/day) so a real
#: policy never hits them; they exist so ``max_alerts_per_hour: 1e9`` is refused.
_BUDGET_LIMITS = {
    "max_alerts_per_hour": 120,
    "max_tokens_per_alert": 5_000_000,
    "max_tokens_per_day": 50_000_000,
}
_MAX_CONCURRENT_CLAIMS = 10
_MAX_LIST = 64
#: Room for well over a hundred channel entries; small enough that parsing a damaged
#: or hostile file stays cheap on every alert (the file is re-read per wake).
_MAX_FILE_BYTES = 64 * 1024

_TOP_KEYS = frozenset({"version", "responders"})
_CHANNEL_KEYS = frozenset(
    {
        "enabled",
        "consent_version",
        "consent_sha256",
        "autonomy_tier",
        "min_severity",
        "schedule",
        "allowed_app_ids",
        "allowed_hook_ids",
        "paused_until",
        "max_concurrent_claims",
        "credential_profile",
        "allowed_repos",
        "allowed_runbooks",
        "allowed_link_hosts",
        "budgets",
    }
)


class PolicyError(ValueError):
    """The file is present but not a valid policy: every channel is OFF."""


@dataclass(frozen=True)
class Budgets:
    max_alerts_per_hour: int = 6
    max_tokens_per_alert: int = 200_000
    max_tokens_per_day: int = 2_000_000


@dataclass(frozen=True)
class ChannelPolicy:
    """One channel's standing authorization, as the owner wrote it."""

    conversation_id: str
    enabled: bool
    consent_version: str
    consent_sha256: str
    autonomy_tier: str
    min_severity: str
    allowed_app_ids: frozenset[str] = frozenset()
    allowed_hook_ids: frozenset[str] = frozenset()
    paused_until: _dt.datetime | None = None
    max_concurrent_claims: int = 1
    credential_profile: str | None = None
    allowed_repos: tuple[str, ...] = ()
    allowed_runbooks: tuple[str, ...] = ()
    allowed_link_hosts: tuple[str, ...] = ()
    budgets: Budgets = field(default_factory=Budgets)

    def paused(self, now: _dt.datetime) -> bool:
        return self.paused_until is not None and now < self.paused_until

    @property
    def may_claim(self) -> bool:
        return self.autonomy_tier in CLAIMING_TIERS


@dataclass(frozen=True)
class ResponderPolicy:
    """The whole parsed file. ``channels`` is empty when the file is absent or invalid."""

    channels: Mapping[str, ChannelPolicy]
    sha256: str | None
    #: Why the policy is empty when it is: ``absent`` | ``invalid: <reason>``. Developer
    #: diagnostics only (logged by the runner), never shown as product copy.
    problem: str | None = None

    def channel(self, conversation_id: str) -> ChannelPolicy | None:
        return self.channels.get(conversation_id)


OFF = ResponderPolicy(channels={}, sha256=None, problem="absent")


def policy_path(config_dir: Path | str) -> Path:
    """Where the owner's policy lives. Also named by the guardrail evidence class."""

    return Path(config_dir) / "team_bridge" / "responders.yaml"


def policy_sha256(raw: bytes) -> str:
    """The hash the hub records as ``policy_sha256``: of the exact bytes on disk."""

    return hashlib.sha256(raw).hexdigest()


def severity_rank(severity: str) -> int | None:
    return _SEVERITY_RANK.get(severity)


def _ids(value: Any, pattern: re.Pattern[str], where: str) -> frozenset[str]:
    if value is None:
        return frozenset()
    if not isinstance(value, list) or len(value) > _MAX_LIST:
        raise PolicyError(f"{where} must be a list of at most {_MAX_LIST} ids")
    out = set()
    for item in value:
        if not isinstance(item, str) or pattern.fullmatch(item) is None:
            raise PolicyError(f"{where} has an invalid id")
        out.add(item)
    return frozenset(out)


def _strings(value: Any, pattern: re.Pattern[str] | None, where: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list) or len(value) > _MAX_LIST:
        raise PolicyError(f"{where} must be a list of at most {_MAX_LIST} entries")
    out: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip() or len(item) > 256:
            raise PolicyError(f"{where} has an invalid entry")
        if pattern is not None and pattern.fullmatch(item) is None:
            raise PolicyError(f"{where} has an invalid entry")
        out.append(item)
    return tuple(out)


def _bounded_int(value: Any, where: str, *, default: int, upper: int) -> int:
    if value is None:
        return default
    # ``bool`` is an ``int`` subclass: ``true`` must not read as 1.
    if not isinstance(value, int) or isinstance(value, bool) or not 1 <= value <= upper:
        raise PolicyError(f"{where} must be an integer between 1 and {upper}")
    return value


def _timestamp(value: Any, where: str) -> _dt.datetime | None:
    if value is None:
        return None
    if isinstance(value, _dt.datetime):
        parsed = value
    elif isinstance(value, str):
        try:
            parsed = _dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            raise PolicyError(f"{where} is not an ISO-8601 timestamp") from None
    else:
        raise PolicyError(f"{where} is not an ISO-8601 timestamp")
    if parsed.tzinfo is None:
        # A naive time is ambiguous across the owner's machines and the hub: refuse.
        raise PolicyError(f"{where} must carry a timezone")
    return parsed.astimezone(_dt.timezone.utc)


def _channel(conversation_id: str, raw: Any) -> ChannelPolicy:
    where = f"responders.{conversation_id}"
    if not isinstance(raw, Mapping):
        raise PolicyError(f"{where} must be a mapping")
    unknown = set(raw) - _CHANNEL_KEYS
    if unknown:
        # An unknown key is most likely a misspelt limit; ignoring it would widen authority.
        raise PolicyError(f"{where} has unknown keys: {', '.join(sorted(map(str, unknown)))}")
    enabled = raw.get("enabled", False)
    if not isinstance(enabled, bool):
        raise PolicyError(f"{where}.enabled must be true or false")
    consent_version = raw.get("consent_version")
    if consent_version != CONSENT_VERSION:
        raise PolicyError(f"{where}.consent_version must be {CONSENT_VERSION}")
    consent_sha = raw.get("consent_sha256")
    if not isinstance(consent_sha, str) or _SHA256.fullmatch(consent_sha) is None:
        raise PolicyError(f"{where}.consent_sha256 must be the 64-hex hash of the served consent")
    tier = raw.get("autonomy_tier", "T0")
    if tier not in AUTONOMY_TIERS:
        raise PolicyError(f"{where}.autonomy_tier must be one of {', '.join(AUTONOMY_TIERS)}")
    min_severity = raw.get("min_severity", "sev2")
    if min_severity not in _SEVERITY_RANK:
        raise PolicyError(f"{where}.min_severity must be sev1..sev4")
    schedule = raw.get("schedule", "always")
    if schedule != "always":
        raise PolicyError(f"{where}.schedule: only 'always' is supported by this build")
    budgets_raw = raw.get("budgets") or {}
    if not isinstance(budgets_raw, Mapping) or set(budgets_raw) - set(_BUDGET_LIMITS):
        raise PolicyError(f"{where}.budgets has unknown keys or is not a mapping")
    defaults = Budgets()
    budgets = Budgets(
        **{
            key: _bounded_int(
                budgets_raw.get(key),
                f"{where}.budgets.{key}",
                default=getattr(defaults, key),
                upper=upper,
            )
            for key, upper in _BUDGET_LIMITS.items()
        }
    )
    profile = raw.get("credential_profile")
    if profile is not None and (
        not isinstance(profile, str) or _PROFILE.fullmatch(profile) is None
    ):
        raise PolicyError(f"{where}.credential_profile is not a profile name")
    return ChannelPolicy(
        conversation_id=conversation_id,
        enabled=enabled,
        consent_version=consent_version,
        consent_sha256=consent_sha,
        autonomy_tier=tier,
        min_severity=min_severity,
        allowed_app_ids=_ids(raw.get("allowed_app_ids"), _APP_ID, f"{where}.allowed_app_ids"),
        allowed_hook_ids=_ids(raw.get("allowed_hook_ids"), _HOOK_ID, f"{where}.allowed_hook_ids"),
        paused_until=_timestamp(raw.get("paused_until"), f"{where}.paused_until"),
        max_concurrent_claims=_bounded_int(
            raw.get("max_concurrent_claims"),
            f"{where}.max_concurrent_claims",
            default=1,
            upper=_MAX_CONCURRENT_CLAIMS,
        ),
        credential_profile=profile,
        allowed_repos=_strings(raw.get("allowed_repos"), _REPO, f"{where}.allowed_repos"),
        allowed_runbooks=_strings(raw.get("allowed_runbooks"), None, f"{where}.allowed_runbooks"),
        allowed_link_hosts=_strings(
            raw.get("allowed_link_hosts"), _HOST, f"{where}.allowed_link_hosts"
        ),
        budgets=budgets,
    )


class _DuplicateKey(Exception):
    def __init__(self, key: object) -> None:
        super().__init__(key)
        self.key = key


def _unique_key_loader() -> type:
    """A ``SafeLoader`` that refuses a repeated mapping key.

    Why (review m2): ``safe_load`` keeps the LAST duplicate, so ``min_severity:
    sev1`` … ``min_severity: sev4`` parses as sev4 — exactly the silent widening
    the all-or-nothing parse exists to rule out. Built per call so the subclass
    never leaks a constructor into the shared ``SafeLoader``.
    """

    import yaml

    class _Loader(yaml.SafeLoader):
        pass

    def _mapping(
        loader: yaml.SafeLoader, node: yaml.MappingNode, deep: bool = False
    ) -> dict[Any, Any]:
        loader.flatten_mapping(node)
        seen: set[object] = set()
        for key_node, _value in node.value:
            key = loader.construct_object(key_node, deep=True)
            try:
                duplicate = key in seen
            except TypeError:  # an unhashable key: the shape check refuses it later
                continue
            if duplicate:
                raise _DuplicateKey(key)
            seen.add(key)
        return loader.construct_mapping(node, deep=deep)

    _Loader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _mapping)
    return _Loader


def parse_policy(raw: bytes) -> ResponderPolicy:
    """Parse the file's bytes. Raises :class:`PolicyError`; :func:`load_policy` maps it to OFF."""

    import yaml

    if len(raw) > _MAX_FILE_BYTES:
        raise PolicyError("the policy file is larger than 64 KiB")
    try:
        doc = yaml.load(raw.decode("utf-8"), Loader=_unique_key_loader())  # noqa: S506
    except _DuplicateKey as exc:
        raise PolicyError(f"the policy repeats the key {exc.key!r}") from None
    except (UnicodeDecodeError, yaml.YAMLError):
        raise PolicyError("the policy file is not valid UTF-8 YAML") from None
    if not isinstance(doc, Mapping) or set(doc) - _TOP_KEYS:
        raise PolicyError("the policy must be a mapping with only 'version' and 'responders'")
    if doc.get("version") != 1:
        raise PolicyError("the policy needs 'version: 1'")
    responders = doc.get("responders") or {}
    if not isinstance(responders, Mapping):
        raise PolicyError("'responders' must be a mapping of conversation ids")
    channels: dict[str, ChannelPolicy] = {}
    for conversation_id, entry in responders.items():
        if (
            not isinstance(conversation_id, str)
            or _CONVERSATION_ID.fullmatch(conversation_id) is None
        ):
            raise PolicyError("a responders key is not a bcv_ conversation id")
        channels[conversation_id] = _channel(conversation_id, entry)
    return ResponderPolicy(channels=channels, sha256=policy_sha256(raw))


def _open_dir_nofollow(name: str | Path, what: str, *, dir_fd: int | None = None) -> int:
    """Open one directory level without following a link at that level."""

    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        return os.open(name, flags, dir_fd=dir_fd)
    except OSError as exc:
        # A link under O_NOFOLLOW is ELOOP on Linux; with O_DIRECTORY darwin may
        # answer ENOTDIR instead. Either way it is not the directory we vetted.
        if exc.errno in (errno.ELOOP, errno.ENOTDIR):
            raise PolicyError(f"{what} is a symlink or not a directory") from None
        raise


def _owned_and_private(st: os.stat_result, what: str) -> None:
    """Owned by this user and not writable by group or world: otherwise another
    principal (or a planted copy carrying foreign bits) chose these bytes."""

    if hasattr(os, "getuid") and st.st_uid != os.getuid():
        raise PolicyError(f"{what} is not owned by this user")
    if st.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
        raise PolicyError(f"{what} is group- or world-writable")


def _read_fenced(path: Path) -> bytes:
    """The policy's bytes, refusing every shape an agent could have planted.

    WHY THE LOADER, NOT ONLY THE CLASSIFIER (PR #4383 security rounds 1-2: M1/S-1,
    S-11, n-2). Two rounds showed that enumerating write commands in the guardrail
    classifier keeps losing a route (``cp -l``, ``rsync`` into the state root, an
    interpreter's ``os.link``). So the loader proves, on the descriptors it
    actually reads, that the bytes can only have come from the owner writing this
    one name:

    * the state directory and ``team_bridge`` are each opened ``O_DIRECTORY |
      O_NOFOLLOW`` and walked by ``dir_fd`` (``openat``), so the directory that
      was checked is the directory read — no lstat-then-open-by-path window, and
      no symlinked level (a linked state root is refused too: the owner keeps it
      a real directory, and a link there would hand the policy to whoever owns
      the link target);
    * both directories must be owned by this uid and not group/world-writable;
    * the file is opened ``O_NOFOLLOW`` relative to ``team_bridge`` and ``fstat``
      on that fd must show a regular file, owned by this uid, not
      group/world-writable, and ``st_nlink == 1``. The link count is the S-11
      closure: a hardlink (``ln``, ``cp -l``, ``cp -al``, ``os.link``) gives the
      policy inode a second, agent-writable name, and no command classifier can
      see every way to make one; a policy with another name is not a policy.

    ``FileNotFoundError`` at any level is "no policy" (OFF); every other refusal
    is :class:`PolicyError` → OFF with a problem the owner can read.
    """

    team_bridge = path.parent
    state = team_bridge.parent
    fds: list[int] = []
    try:
        fds.append(_open_dir_nofollow(state, "the state directory"))
        st = os.fstat(fds[-1])
        _owned_and_private(st, "the state directory")
        fds.append(_open_dir_nofollow(team_bridge.name, "team_bridge", dir_fd=fds[-1]))
        st = os.fstat(fds[-1])
        _owned_and_private(st, "the team_bridge directory")
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
        try:
            fd = os.open(path.name, flags, dir_fd=fds[-1])
        except OSError as exc:
            # O_NOFOLLOW on a link (dangling or not) is ELOOP on darwin and Linux.
            if exc.errno == errno.ELOOP:
                raise PolicyError("the policy file is a symlink") from None
            raise
        fds.append(fd)
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode):
            raise PolicyError("the policy is not a regular file")
        if st.st_nlink != 1:
            raise PolicyError("the policy file has another name (hardlink)")
        _owned_and_private(st, "the policy file")
        if st.st_size > _MAX_FILE_BYTES:
            raise PolicyError("the policy file is larger than 64 KiB")
        with os.fdopen(fds.pop(), "rb", closefd=True) as handle:
            return handle.read(_MAX_FILE_BYTES + 1)
    finally:
        for fd in fds:
            os.close(fd)


def load_policy(config_dir: Path | str) -> ResponderPolicy:
    """Read the policy NOW. Absent or invalid -> :data:`OFF`-shaped (no channel enabled)."""

    path = policy_path(config_dir)
    try:
        raw = _read_fenced(path)
    except FileNotFoundError:
        return OFF
    except PolicyError as exc:
        return ResponderPolicy(channels={}, sha256=None, problem=f"invalid: {exc}")
    except OSError as exc:
        return ResponderPolicy(
            channels={}, sha256=None, problem=f"invalid: unreadable ({exc.__class__.__name__})"
        )
    try:
        return parse_policy(raw)
    except PolicyError as exc:
        return ResponderPolicy(channels={}, sha256=None, problem=f"invalid: {exc}")


@dataclass(frozen=True)
class Admission:
    """The local verdict on one alert wake. ``reason`` is a closed developer code."""

    admitted: bool
    reason: str
    channel: ChannelPolicy | None = None


def admit_alert(
    policy: ResponderPolicy,
    *,
    conversation_id: str,
    severity: str,
    app_id: str | None,
    hook_id: str | None,
    now: _dt.datetime,
    alerts_last_hour: int,
    active_claims: int,
) -> Admission:
    """May this alert wake turn into work on this device? Every miss is a refusal.

    The hub already filtered by its own copy of enable/pause/consent and by the
    installation's ``wake_min_severity``; this re-checks against the LOCAL policy,
    because the hub's copy is a projection the owner's device does not trust to be
    current (a local pause must bite even when the PUT that mirrors it failed).
    """

    channel = policy.channel(conversation_id)
    if channel is None:
        return Admission(False, "no_policy")
    if not channel.enabled:
        return Admission(False, "disabled", channel)
    if channel.paused(now):
        return Admission(False, "paused", channel)
    rank = severity_rank(severity)
    if rank is None or rank > _SEVERITY_RANK[channel.min_severity]:
        return Admission(False, "below_min_severity", channel)
    if channel.allowed_app_ids and app_id not in channel.allowed_app_ids:
        return Admission(False, "app_not_allowed", channel)
    if channel.allowed_hook_ids and hook_id not in channel.allowed_hook_ids:
        return Admission(False, "hook_not_allowed", channel)
    if alerts_last_hour >= channel.budgets.max_alerts_per_hour:
        return Admission(False, "hourly_budget", channel)
    if active_claims >= channel.max_concurrent_claims:
        return Admission(False, "concurrency", channel)
    return Admission(True, "admitted", channel)
