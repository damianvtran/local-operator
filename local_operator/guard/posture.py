"""Posture: binarised weekly facts under memoised randomised response (RFC v0.9 §10').

WHAT. Six yes/no facts about how this install's guardrails are set up, each
reduced to ``recommended`` / ``other`` and then REPORTED ONLY AS A
RANDOMISED BIT. The true value never leaves the computer.

THE MEMO (the load-bearing privacy property, §10' "On the install (L3)").
For fact ``f`` with true value ``v``, :class:`PostureMemo` keeps one bit per
``(account, f, v)``, drawn ONCE the first time ``v`` is reported::

    b = [v is recommended]  with probability p = e^ε / (1 + e^ε) = 0.75 (ε = ln 3)
    b = its complement      otherwise

and reports that same ``b`` for as long as the true value is ``v``. Returning
to an earlier value reuses its memo -- never a fresh draw -- so a binary fact
exposes at most two bits for life however often it flips. A fresh draw per
change would let the leakage grow with the number of changes (the RFC's
``redraw_negative_control``).

WHAT NEVER RESETS IT. The memo is keyed on the ACCOUNT, not the tenant (two
orgs see the same bit and cannot average independent draws), and lives in its
own file, ``<config_dir>/guard/posture_memo.sqlite3``, which no revoke,
re-grant, org off->on, plan gap, widening, rejoin or ``namespace_reset`` path
touches (``guard/producers.py`` purges queues, never this file). Only the
member deleting the install's Guard state draws again (the RFC's
member-controlled-redraw residual).

WHEN IT IS SENT. The OPEN week's memo is reported during the week and sent
again whenever a reported bit changes; the last report of a week wins on the
server (A3 ``docs/GUARD.md`` "Deltas for lop" #6). It is never held back until
the week closes: the server stops taking a week 24 h after its Sunday, so a
laptop that was off that weekend would otherwise lose the whole week
(:func:`week_is_open`).

ONE DESIGNATED DEVICE PER MEMBER is the server's job (A3's persisted
``posture_designation``); this install simply reports under a live
``posture`` grant and the server ignores it when it is not the designated one.

WHAT A FACT READS. lop's own config values and its own caches, never a scan
of the member's files: ``guardrails.enabled`` / ``guardrails.critical_override``
as written in ``config.yml``, a COUNT of rule files in lop's rules folders,
the packaged default policy's digest, and the update cache's age and version
(``update.cached_latest``, which never fetches). A fact this install cannot
measure is omitted from the batch -- "unsupported is incomplete, not zero".
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import math
import random
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from ._sqlite import open_store

POSTURE_SCHEMA: Final = "guard_posture_v1"
MEMO_SCHEMA_VERSION: Final = 1

#: Frozen in grid v2 (RFC §10'): ε = ln 3, p = e^ε / (1 + e^ε).
EPSILON: Final = math.log(3.0)
P_TRUTH: Final = math.exp(EPSILON) / (1.0 + math.exp(EPSILON))  # 0.75

VALUE_RECOMMENDED: Final = "recommended"
VALUE_OTHER: Final = "other"
VALUES: Final = frozenset({VALUE_RECOMMENDED, VALUE_OTHER})

#: The six posture families (RFC §10': "six separate families, one per
#: posture fact"), v0.3 §8's facts binarised to recommended/other. The
#: recommended value of each is stated beside it so A3's grid and Console copy
#: can pin the same reading.
FACT_GUARDRAILS_ENABLED: Final = "guardrails_enabled"  # recommended: user rules on
FACT_CRITICAL_OVERRIDE_OFF: Final = "critical_override_off"  # recommended: override not set
FACT_USER_RULES_PRESENT: Final = "user_rules_present"  # recommended: >= 1 user rule
FACT_BUILTIN_POLICY_INTACT: Final = "builtin_policy_intact"  # recommended: shipped default
FACT_VERSION_CURRENT: Final = "guard_version_current"  # recommended: newest known release
FACT_UPDATE_CHECK_RECENT: Final = "update_check_recent"  # recommended: checked <= 7 days
FACTS: Final = (
    FACT_GUARDRAILS_ENABLED,
    FACT_CRITICAL_OVERRIDE_OFF,
    FACT_USER_RULES_PRESENT,
    FACT_BUILTIN_POLICY_INTACT,
    FACT_VERSION_CURRENT,
    FACT_UPDATE_CHECK_RECENT,
)

#: v0.3 §8 ``last_check_band``'s first band, as the binarisation threshold.
UPDATE_CHECK_RECENT_S: Final = 7 * 24 * 3600.0


def memo_path(config_dir: Path | str) -> Path:
    return Path(config_dir) / "guard" / "posture_memo.sqlite3"


_MEMO_DDL: Final = (
    "CREATE TABLE memo (account_key TEXT NOT NULL, fact TEXT NOT NULL, value TEXT NOT NULL,"
    " bit INTEGER NOT NULL CHECK (bit IN (0, 1)), PRIMARY KEY (account_key, fact, value))",
)


def account_key(account_id: str) -> str:
    """The memo's key for an account. Local only; hashed so the file names no account."""

    return hashlib.sha256(f"radient-guard-posture-memo-v1:{account_id}".encode()).hexdigest()


class PostureMemo:
    """The per-(account, fact, value) memo. Construction does no I/O."""

    def __init__(self, config_dir: Path | str, *, rng: Callable[[], float] | None = None) -> None:
        self._path = memo_path(config_dir)
        # SystemRandom: the bit must not be predictable from anything else on
        # the machine (a seeded PRNG would let a reader of other state guess it).
        self._rng = rng or random.SystemRandom().random

    @property
    def path(self) -> Path:
        return self._path

    def bit(self, account: str, fact: str, value: str) -> int:
        """The memoised randomised answer for ``value``; drawn only on first sight."""

        if fact not in FACTS or value not in VALUES:
            raise ValueError("unknown posture fact or value")
        key = account_key(account)
        conn = open_store(self._path, version=MEMO_SCHEMA_VERSION, ddl=_MEMO_DDL)
        try:
            conn.execute("BEGIN IMMEDIATE")
            found = conn.execute(
                "SELECT bit FROM memo WHERE account_key = ? AND fact = ? AND value = ?",
                (key, fact, value),
            ).fetchone()
            if found is not None:
                conn.execute("COMMIT")
                return int(found["bit"])
            truthful = 1 if value == VALUE_RECOMMENDED else 0
            drawn = truthful if self._rng() < P_TRUTH else 1 - truthful
            conn.execute(
                "INSERT INTO memo (account_key, fact, value, bit) VALUES (?, ?, ?, ?)",
                (key, fact, value, drawn),
            )
            conn.execute("COMMIT")
            return drawn
        except BaseException:
            if conn.in_transaction:
                conn.execute("ROLLBACK")
            raise
        finally:
            conn.close()

    def memo_count(self, account: str) -> int:
        if not self._path.exists():
            return 0
        conn = open_store(self._path, version=MEMO_SCHEMA_VERSION, ddl=_MEMO_DDL)
        try:
            return int(
                conn.execute(
                    "SELECT COUNT(*) FROM memo WHERE account_key = ?", (account_key(account),)
                ).fetchone()[0]
            )
        finally:
            conn.close()


@dataclass(frozen=True)
class PostureReport:
    """One week's posture answer for one fact: the BIT only."""

    fact: str
    bit: int
    week: str  # ISO week, e.g. ``2026-W48``

    def wire(self) -> dict[str, object]:
        return {"schema": POSTURE_SCHEMA, "fact": self.fact, "bit": self.bit, "week": self.week}


def iso_week(day: _dt.date) -> str:
    year, week, _ = day.isocalendar()
    return f"{year}-W{week:02d}"


#: How long after its Sunday a posture week still counts server-side: A3's
#: ``guard.PostureCloseLag`` (24 h), kept at one day so the retained bit never
#: outlives RFC v0.9 §10' S-2b's 8-day window. Deliberately NOT lop's 72 h
#: queue TTL for day records: a posture week lop finds closed past this is lost
#: to the server, so the producers send the OPEN week during the week instead.
POSTURE_CLOSE_LAG_DAYS: Final = 1


def week_is_open(week: str, today: _dt.date) -> bool:
    """Whether the server still accepts ``week``'s memo on UTC day ``today``.

    Open from the week's Monday until the end of the Monday after it (week end
    + :data:`POSTURE_CLOSE_LAG_DAYS`). Judged on the UTC DATE, like the server,
    so a clock a few minutes off cannot move the verdict except at midnight.
    A malformed week is never open.
    """

    try:
        year, number = week.split("-W")
        monday = _dt.date.fromisocalendar(int(year), int(number), 1)
    except ValueError:
        return False
    return today < monday + _dt.timedelta(days=7 + POSTURE_CLOSE_LAG_DAYS)


def _digest(items: list[str]) -> str:
    material = "\x00".join(sorted(items)).encode()
    return hashlib.sha256(b"radient-guard-posture-sent-v1\x00" + material).hexdigest()[:32]


def bits_digest(reports: Mapping[str, int] | list[PostureReport]) -> str:
    """Digest of one week's REPORTED bits: what the server holds after a send.

    Equal digests mean the server already has this exact memo for the week, so
    re-sending would add nothing. Bits only, no true value.
    """

    if isinstance(reports, Mapping):
        return _digest([f"{fact}={bit}" for fact, bit in reports.items()])
    return _digest([f"{r.fact}={r.bit}" for r in reports])


def values_digest(values: Mapping[str, str]) -> str:
    """Digest of one week's TRUE values. Local only (``producers.sqlite3``).

    It exists so an idle tick can tell "nothing changed" without resolving the
    sign-in or reading the memo: the memo maps a value to its bit
    deterministically, so equal values give equal bits.
    """

    return _digest([f"{fact}={value}" for fact, value in values.items()])


# ---------------------------------------------------------------------------
# Reading the facts (true values; never leave this module except as a memo key)
# ---------------------------------------------------------------------------

#: SHA-256 of the packaged ``guardrails/defaults/dangerous-delete.md``. A test
#: pins this against the file, so a deliberate edit to the shipped default
#: updates the digest in the same change; any other difference -- an edited
#: install tree -- reads as "not intact".
BUILTIN_POLICY_SHA256: Final = "b85c0fc1231e621711b627e1f24d906e0c5f949224c71c906932fcac8307389c"


def _builtin_digest() -> str | None:
    try:
        path = Path(__file__).resolve().parent.parent / "guardrails" / "defaults"
        files = sorted(p for p in path.iterdir() if p.is_file() and p.suffix == ".md")
        digest = hashlib.sha256()
        for item in files:
            digest.update(item.name.encode() + b"\x00" + item.read_bytes() + b"\x00")
        return digest.hexdigest()
    except OSError:
        return None


def _binary(ok: bool | None) -> str | None:
    if ok is None:
        return None
    return VALUE_RECOMMENDED if ok else VALUE_OTHER


def read_facts(config_dir: Path | str) -> dict[str, str]:
    """``{fact: recommended|other}`` for every fact this install can measure. Never raises."""

    facts: dict[str, str | None] = {}
    config_dir = Path(config_dir)
    try:
        from local_operator.config import ConfigManager
        from local_operator.guardrails import check as guard_check

        config = ConfigManager(config_dir)
        enabled = config.get_nested_value(
            guard_check.GUARDRAILS_ENABLED_PATH, guard_check.GUARDRAILS_ENABLED_DEFAULT
        )
        override = config.get_nested_value(
            guard_check.GUARDRAILS_CRITICAL_OVERRIDE_PATH,
            guard_check.GUARDRAILS_CRITICAL_OVERRIDE_DEFAULT,
        )
        facts[FACT_GUARDRAILS_ENABLED] = _binary(enabled is not False)
        facts[FACT_CRITICAL_OVERRIDE_OFF] = _binary(override is not True)
        rules_raw = config.get_nested_value(
            guard_check.GUARDRAILS_RULES_DIR_PATH, guard_check.GUARDRAILS_RULES_DIR_DEFAULT
        )
        dirs = [config_dir / "rules"]
        if isinstance(rules_raw, str) and rules_raw:
            dirs.append(Path(rules_raw).expanduser())
        count = 0
        for folder in dirs:
            if folder.is_dir():
                # A COUNT of rule files, never their names or contents.
                count += sum(1 for p in folder.iterdir() if p.is_file() and p.suffix == ".md")
        facts[FACT_USER_RULES_PRESENT] = _binary(count >= 1)
    except Exception:  # noqa: BLE001 -- an unreadable config is an unmeasured fact
        pass
    digest = _builtin_digest()
    facts[FACT_BUILTIN_POLICY_INTACT] = _binary(
        None if digest is None else digest == BUILTIN_POLICY_SHA256
    )
    try:
        from local_operator import update

        latest, age = update.cached_latest()
        installed = _installed_version()
        facts[FACT_VERSION_CURRENT] = _binary(
            None if latest is None or installed is None else installed == latest
        )
        facts[FACT_UPDATE_CHECK_RECENT] = _binary(
            None if age is None else age <= UPDATE_CHECK_RECENT_S
        )
    except Exception:  # noqa: BLE001
        pass
    return {k: v for k, v in facts.items() if v is not None}


def _installed_version() -> str | None:
    try:
        from importlib.metadata import version

        return version("local-operator")
    except Exception:  # noqa: BLE001
        return None


def reports_for(
    memo: PostureMemo, account: str, week: str, values: Mapping[str, str]
) -> list[PostureReport]:
    """The batched posture for ``week``: one memoised bit per measured fact."""

    return [
        PostureReport(fact=fact, bit=memo.bit(account, fact, value), week=week)
        for fact, value in sorted(values.items())
        if fact in FACTS and value in VALUES
    ]


__all__ = [
    "BUILTIN_POLICY_SHA256",
    "EPSILON",
    "FACTS",
    "P_TRUTH",
    "POSTURE_CLOSE_LAG_DAYS",
    "POSTURE_SCHEMA",
    "PostureMemo",
    "PostureReport",
    "VALUE_OTHER",
    "VALUE_RECOMMENDED",
    "account_key",
    "bits_digest",
    "iso_week",
    "memo_path",
    "read_facts",
    "reports_for",
    "values_digest",
    "week_is_open",
]
