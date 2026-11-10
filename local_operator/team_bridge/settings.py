"""The ``team_bridge.*`` configuration keys, read the way the ``/settings`` registry writes them.

WHY THE DEFAULTS LIVE HERE. ``settings_io`` keeps this package off its import
path (the registry is loaded on every CLI start and on every ``lop serve``
boot), so the registry carries LITERAL defaults and
``tests/unit/test_settings_io.py::_consumer_defaults`` imports THESE constants
to pin that the two cannot drift. The readers sit next to the constants for the
same reason: one place says what "unset" means.

WHY ``get_nested_value``. Every key here is a genuinely nested tuple
(``("team_bridge", "enabled")``); the flat ``ConfigManager`` accessor looks a
dotted string up as ONE literal top-level key (the ``display.*`` exception),
so passing it ``team_bridge.enabled`` would silently read nothing and the
feature would look configured while never following the setting.

SCOPE. These are the LOCAL, per-machine keys. Eligibility, consent and the
endpoint state are SERVER state (``GET /v1/messaging/status`` and
``/settings``); nothing here duplicates them. The master switch is a machine
gate — the composed gate a surface checks is capability × eligible × enabled
(RFC §5.4). Each reader's fallback is deliberate: an ABSENT value reads the
shipped default (OFF for ``enabled`` during the consent hold), while a PRESENT
but unparseable value reads what the shipped default says, never a coerced ON.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Mapping

from local_operator.team_bridge.consent_versions import (
    CONSENT_VERSION_V2,
    DEFAULT_ON_CONSENT_VERSIONS,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from local_operator.config import ConfigManager

#: ``team_bridge.enabled`` — OFF by default (a consent hold).
#:
#: #2712 shipped the opt-out default ON, but the server still served
#: ``bridge_consent_v1``, whose copy describes the data handling without ever
#: offering an opt-out. A status read could therefore turn an eligible Team
#: plan on and write a consent receipt before the member saw anything that
#: asked them. Until a consent version carrying opt-out copy is served AND
#: accepted, an absent key must mean OFF. This key remains only the LOCAL half
#: of the composed gate — capability × eligible × enabled (RFC §5.4) — so an
#: explicit ``true`` (the settings row, a hand edit) enables only the surfaces on
#: this machine; the server still decides eligibility, consent and launch.
#: Restoring the ON default belongs to the new-consent-version change, not a
#: local flip.
#:
#: DEFAULT-ON UNDER ``bridge_consent_v2``/``v3`` DOES NOT CHANGE THIS CONSTANT. The
#: registry literal and every consumer that reads ``enabled`` from disk (host,
#: read_context, authority, wake, engagement) keep reading an absent key as OFF,
#: so a hub that serves only v1 is byte-identical to the hold. The opt-out
#: default is a COMPOSED gate that needs a fact this module cannot know — the
#: hub's served consent — and so lives in
#: :meth:`TeamBridgeSettings.enabled_given`, used by the status discipline (the
#: only automatic enrolment) and the status route's ``local.enabled``. The
#: automatic enrolment then WRITES ``team_bridge.enabled = true`` once the owner
#: and receipt are committed (:func:`record_local_enabled_if_unset`), which is what makes
#: the offline consumers above agree; an absent key therefore never starts any
#: send/receive/wake work by itself — it only means "eligible to be enrolled".
DEFAULT_TEAM_BRIDGE_ENABLED = False

#: The consent versions whose copy discloses the opt-out default are
#: :data:`local_operator.team_bridge.consent_versions.DEFAULT_ON_CONSENT_VERSIONS`
#: (a leaf the pure reconcile planner may import too).

#: ``team_bridge.onboarding`` — the one-time dialog's shown-flag (RFC §5.3:
#: client-local in v1, deliberately never a server row). ``""`` means "not
#: dismissed yet"; :data:`ONBOARDING_SEEN` means the dialog was dismissed on
#: this machine. An unknown value normalises to ``""`` — for a UX flag the
#: safe direction is to allow the one-time notice to show again, never to
#: suppress it forever.
DEFAULT_TEAM_BRIDGE_ONBOARDING = ""
ONBOARDING_SEEN = "seen"

#: ``team_bridge.default_enrollment`` — internal state (deliberately NOT a
#: registry row, exactly like ``onboarding``): a mapping of ACCOUNT id to
#: ``"done"`` recording the consent write an earlier opt-out default ran for
#: that member. PER ACCOUNT on purpose: it was a per-member product rule, so
#: account A enrolling did not suppress account B on the same machine.
#:
#: WHO WRITES IT. Under ``bridge_consent_v1`` nothing does (consent hold,
#: #2712: a status read must never enrol). Under a default-on consent
#: (``bridge_consent_v2``/``v3``, ``consent_versions.DEFAULT_ON_CONSENT_VERSIONS``) two
#: places do, and both mean "never enrol this account from this machine again":
#: the gated automatic enrolment (``StatusDiscipline._default_enroll``) after the
#: server created the endpoint, and the one-click turn-off (a successful
#: ``state=disabled`` write). It is MEMORY that suppresses a repeat, never
#: authority: no write is ever performed BECAUSE this key says so, and a missing
#: key never proves anything either (the server's own no-op-when-an-endpoint-
#: exists rule is the authoritative stickiness).
#:
#: ``pending:<endpoint_id>:<head>:<consent_version>`` (D1, review R3/R2-1; the
#: version since #3225, see :func:`parse_pending_default_enrollment`) is the one other
#: value: THIS machine's default-on POST created ``endpoint_id`` but the local
#: owner commit failed or was vetoed. Without it the account was marked
#: ``done`` from the hub's answer alone, the surface then read state C, and no
#: path ever bound the device ("on, but deaf"). The status discipline retries
#: ONLY the local commit against that exact endpoint — never a second POST —
#: seeded at ``head``, the feed head read BEFORE the original POST (``0`` when
#: there was none, a brand-new endpoint), so a message that arrived while the
#: device could not receive is never skipped. It still suppresses a repeat
#: POST exactly like ``done``; an explicit OFF or a Turn off overwrites it
#: with ``done``.
DEFAULT_TEAM_BRIDGE_ENROLLMENT: Mapping[str, str] = {}
DEFAULT_ENROLLMENT_DONE = "done"
DEFAULT_ENROLLMENT_PENDING_PREFIX = "pending:"


def pending_default_enrollment(endpoint_id: str, head: int | None, consent_version: str) -> str:
    """The ``pending:<endpoint_id>:<head>:<consent_version>`` marker value (see above).

    ``consent_version`` is the version the default-on POST ECHOED — the text the
    member was shown — so the commit-only recovery can refuse a hub record that
    names any other version (review m1 / QA Q2 on #3225).
    """

    if consent_version not in DEFAULT_ON_CONSENT_VERSIONS:
        raise ValueError("a pending marker records a default-on consent version")
    head_part = head if head is not None else 0
    return f"{DEFAULT_ENROLLMENT_PENDING_PREFIX}{endpoint_id}:{head_part}:{consent_version}"


def parse_pending_default_enrollment(value: Any) -> tuple[str, int, str] | None:
    """``(endpoint_id, pre-POST head, echoed consent version)`` from a pending marker.

    Anything malformed reads as "no pending commit" (``None``): the retry is a
    recovery, and a hand-edited value must never point it at an endpoint or a
    cursor. A trailing version outside :data:`DEFAULT_ON_CONSENT_VERSIONS` is
    malformed too (exact membership, never ordering).

    LEGACY MARKERS. A marker written before #3225 is ``pending:<endpoint>:<head>``
    with no version. It reads as ``bridge_consent_v2`` — not a guess: that build's
    ``enroll_default_on`` hardcoded the v2 echo, so v2 is provably what was shown.
    Recovery therefore still binds a pre-#3225 machine whose hub record cites v2,
    and refuses one citing anything else. Reading it as "commit nothing" was the
    alternative; it would strand exactly those machines "on, but deaf" (the D1
    defect the marker exists to repair) for no consent gain.
    """

    if not isinstance(value, str) or not value.startswith(DEFAULT_ENROLLMENT_PENDING_PREFIX):
        return None
    rest = value[len(DEFAULT_ENROLLMENT_PENDING_PREFIX) :]
    prefix, sep, last = rest.rpartition(":")
    if sep and last in DEFAULT_ON_CONSENT_VERSIONS:
        rest, version = prefix, last
    elif last.isdigit():
        version = CONSENT_VERSION_V2
    else:
        return None
    endpoint_id, sep, head = rest.rpartition(":")
    if not sep or not endpoint_id or not head.isdigit() or (len(head) > 1 and head[0] == "0"):
        return None
    return endpoint_id, int(head), version


#: ``team_bridge.session`` — the designated CoS session id (RFC §5.6/§13.6).
#: The connector wakes ONE session per account, and the designation is the
#: consent for it to start a runtime; there is deliberately no list here,
#: because "which sessions may be woken" is not a user-tunable policy in v1
#: and a list would imply one. Empty = no destination; arrivals badge only and
#: never wake a session.
DEFAULT_TEAM_BRIDGE_SESSION = ""

#: ``team_bridge.hold_alert_after_minutes`` — how long a conversation message may
#: sit held (in screening, or admitted and awaiting delivery) before the owner
#: gets ONE local notification (see ``hold_alert.py``). ``0`` turns the alert
#: off; the valid range is 0..10080 (one week).
#:
#: DELIBERATELY NO ``/settings`` REGISTRY ROW. A row would change ``wire.settings``
#: copy, and that edit changes the English source hash of the namespace and
#: un-ships every locale's audited ledger row for it. The key is read from the
#: config file like any other (a hand edit works); a settings row is a later,
#: separately-translated change.
DEFAULT_HOLD_ALERT_AFTER_MINUTES = 60
HOLD_ALERT_MAX_MINUTES = 10080

#: ``team_bridge.hold_release`` — B4b owner release (operator decision a-3f46,
#: reversing a-d9c6): a screening hold whose decision is EXACTLY one work-scope
#: category is surfaced to the receiving person per message, with Deliver and
#: Discard, instead of sitting cached forever behind a count-only alert.
#:
#: OFF BY DEFAULT (B4b DESIGN §8.1 slice S1). With it off, screening, the
#: connector and every notice are byte-identical to the build before B4b; the
#: desktop routes still exist so QA can drive them, and they refuse every
#: write with ``hold_release_off``. Turning it on by default is slice S3a,
#: which waits for the S2 UI and for B5's receiver-adherence results.
#:
#: Read from the config file like ``hold_alert_after_minutes`` and for the same
#: reason has NO ``/settings`` registry row: a row would change
#: ``wire.settings`` copy and un-ship every locale's audited ledger row. Only a
#: real ``true`` turns it on; anything else (absent, ``null``, a string) reads
#: OFF, because a release path must never open on a value nobody wrote.
DEFAULT_HOLD_RELEASE = False

#: ``team_bridge.share_typing`` — whether this person's composers tell the other
#: members of a conversation "X is typing" (typing-indicators design §7).
#:
#: ON BY DEFAULT, like Slack and Discord (neither documents a user toggle);
#: Zulip and Signal offer one, so the person can turn it off. Sender-side only,
#: NOT symmetric: turning it off stops this machine reporting typing and never
#: hides anyone else's indicator. The renderer reads it to skip the heartbeat,
#: and the desktop ``PUT .../typing`` route reads it again (defence in depth)
#: and answers ``{shared: false}`` without contacting the hub.
#:
#: Like ``hold_release`` it has NO ``/settings`` registry row (a row would
#: change ``wire.settings`` copy and un-ship every locale's audited ledger
#: row); the desktop Team Bridge settings routes read and write it, and a hand
#: edit works. An ABSENT key reads ON; a PRESENT key shares only on a real
#: ``true`` — a value somebody wrote that is not ``true`` (``false``, ``null``,
#: a typo) reads OFF, because that writer was trying to change the default and
#: a privacy switch must not reopen on a value it cannot parse.
DEFAULT_TEAM_BRIDGE_SHARE_TYPING = True

#: ``team_bridge.screening.*`` — the rolling-hour ceilings on SCREENING SPEND,
#: now CONFIGURABLE. They were fixed class constants on ``Store`` (40/30/35/20,
#: "the frozen §8 budget"); these are the 10x replacements and the shipped
#: defaults of four ``/settings`` rows.
#:
#: WHY THEY MOVED — MEASURED IN BUDGET UNITS, NOT ROWS. The cap counts DISTINCT
#: ``COALESCE(batch_id, attempt_id)`` per kind, so a RAW ROW count double-counts
#: one screen (a logical row plus an http row, plus shared batch rows). Measured
#: against the live store (``~/.local-operator/team_bridge/state.sqlite3``,
#: read-only, 2026-11-05) in that unit: peak LOGICAL units in one hour = 40 —
#: exactly the old ceiling, reached in four separate hours (2026-10-31 05Z, 07Z,
#: 10Z and 2026-11-05 18Z), with peak HTTP units also 40; the last 24 hours spent
#: 248 logical units against a 40/hour cap. A ceiling that ordinary use touches
#: exactly, in four different hours, is binding — the room spent the hour and the
#: owner's own DM then waited for a refill.
#:
#: HONESTLY: part of that spend was REPEAT screening, not unique demand. 416
#: distinct messages produced 1,044 logical units (2.5 screens per message), 41
#: messages were screened more than twice, 191 logical attempts belong to
#: messages that never got a receipt, and one message was screened 155 times
#: over 54 hours. So the old number was too low for real demand AND was being
#: spent on retries; the raise fixes the first, and the repeat behaviour is worth
#: its own look rather than being hidden by a constant that hid it before.
#:
#: THE TIER RELATIONSHIPS ARE PRESERVED EXACTLY, not rescaled one at a time:
#: ambient 300 of 400 is the old three quarters (30 of 40); unaddressed 350 is
#: the old 35 (everything but a person addressing this agent); sender 200 is
#: half the account ceiling, the old 20-of-40 share, so one sender still cannot
#: monopolise the window however its messages batch. Every reserve is CARVED
#: OUT of the account ceiling, never added to it.
#:
#: A BIGGER CEILING IS NOT A WEAKER SAFETY STORY. The two bounds on a runaway
#: are untouched by these numbers and read none of them: the screening breaker
#: opens on provider FAILURE (its own threshold), and the owner is notified
#: ONCE PER EXHAUSTION EPISODE at whatever the ceiling is. What these keys bound
#: is spend — and an operator who wants it lower can now say so in config, in
#: the running receiver, instead of waiting for a release.
DEFAULT_TEAM_BRIDGE_SCREENING_BUDGET_PER_HOUR = 400
DEFAULT_TEAM_BRIDGE_SCREENING_AMBIENT_BUDGET_PER_HOUR = 300
DEFAULT_TEAM_BRIDGE_SCREENING_UNADDRESSED_BUDGET_PER_HOUR = 350
DEFAULT_TEAM_BRIDGE_SCREENING_SENDER_BUDGET_PER_HOUR = 200

ENABLED_PATH = ("team_bridge", "enabled")
ONBOARDING_PATH = ("team_bridge", "onboarding")
DEFAULT_ENROLLMENT_PATH = ("team_bridge", "default_enrollment")
SESSION_PATH = ("team_bridge", "session")
HOLD_ALERT_AFTER_PATH = ("team_bridge", "hold_alert_after_minutes")
HOLD_RELEASE_PATH = ("team_bridge", "hold_release")
SHARE_TYPING_PATH = ("team_bridge", "share_typing")
#: The SUBTREE the four screening ceilings live under. Readers take this tuple,
#: so no consumer re-spells "where screening lives" and drifts from the writer.
SCREENING_SUBTREE = ("team_bridge", "screening")
#: Leaf names inside :data:`SCREENING_SUBTREE`; :class:`ScreeningBudget` reads
#: exactly these, and the registry's own ``path`` literals spell them (it cannot
#: import this module — see the module docstring).
SCREENING_BUDGET_LEAF = "budget_per_hour"
SCREENING_AMBIENT_BUDGET_LEAF = "ambient_budget_per_hour"
SCREENING_UNADDRESSED_BUDGET_LEAF = "unaddressed_budget_per_hour"
SCREENING_SENDER_BUDGET_LEAF = "sender_budget_per_hour"
#: The largest ceiling the READ path will adopt. It mirrors the ``maximum`` of
#: the four ``/settings`` rows, which this module cannot import (the registry
#: stays off this package's import path — see the module docstring), so
#: ``tests/unit/test_settings_io.py::_consumer_defaults`` pins the two together.
#: The read path clamps DOWN only: an above-maximum hand edit reads as the
#: maximum with a warning rather than as a number no surface could show, edit
#: back or refund.
SCREENING_BUDGET_MAX_READ = 100_000


#: Distinguishes "the key is absent" (the shipped default) from "the key is
#: present and unreadable" (fail closed). ``None`` cannot carry that difference:
#: a hand-edited ``team_bridge: {enabled: null}`` must not read as absent.
_MISSING = object()


def _enabled(value: Any) -> bool:
    """The shipped default, with a present-but-unreadable value reading OFF.

    WHY THIS IS NOT ``_bool(value, DEFAULT)``. ``None`` cannot tell an absent
    key from a hand-edited ``enabled: null``; only an ABSENT key may read the
    shipped default. A real bool is itself, and anything else fails closed —
    the same direction ``read_context`` already takes for a torn read.
    """

    if value is _MISSING:
        return DEFAULT_TEAM_BRIDGE_ENABLED
    return value if isinstance(value, bool) else False


def _onboarding(value: Any) -> str:
    return ONBOARDING_SEEN if value == ONBOARDING_SEEN else DEFAULT_TEAM_BRIDGE_ONBOARDING


def _default_enrollment(value: Any) -> Mapping[str, str]:
    if not isinstance(value, Mapping):
        return {}
    return {
        account: state
        for account, state in value.items()
        if isinstance(account, str)
        and account.strip()
        and (
            state == DEFAULT_ENROLLMENT_DONE or parse_pending_default_enrollment(state) is not None
        )
    }


def _session(value: Any) -> str:
    """One designated session id; anything else reads as "none designated".

    A non-string, or a list left over from an early hand-edit of this key's
    draft shape, reads as unset rather than being coerced: wrongly naming a
    wake destination is worse than naming none (badge-only is the safe state).
    """

    if isinstance(value, str):
        return value.strip()
    return DEFAULT_TEAM_BRIDGE_SESSION


def _screening_ceiling(value: Any, default: int) -> int:
    """A usable ceiling: a real positive int, else the shipped default.

    See :class:`ScreeningBudget` for the whole rule and why nothing here raises.
    """

    if isinstance(value, bool) or not isinstance(value, int):
        return default
    return value if value >= 1 else default


def _walk_subtree(values: Any, path: tuple[str, ...]) -> Any:
    """``values``' node at ``path``, or :data:`_MISSING` when a step is absent.

    The whole-document twin of ``ConfigManager.get_nested_value``: the reader
    that starts from ``config.read_config_values`` has no manager to ask, and a
    step that is not a mapping (a hand-edited ``team_bridge: off``) is an absent
    subtree rather than an error.
    """

    node = values
    for step in path:
        if not isinstance(node, Mapping) or step not in node:
            return _MISSING
        node = node[step]
    return node


def _hold_alert_after_minutes(value: Any) -> int:
    """Whole minutes in 0..10080, else the default.

    A bool is an ``int`` to Python, so ``true`` must be refused by name; a float,
    a string, a negative number or anything past a week reads as the default
    rather than being coerced into a threshold nobody wrote.
    """

    if isinstance(value, bool) or not isinstance(value, int):
        return DEFAULT_HOLD_ALERT_AFTER_MINUTES
    if not 0 <= value <= HOLD_ALERT_MAX_MINUTES:
        return DEFAULT_HOLD_ALERT_AFTER_MINUTES
    return value


@dataclass(frozen=True)
class ScreeningClamp:
    """One configured ceiling the read path held down, and what it reads as.

    The pair is what makes the read's clamp defensible to a person staring at
    the file: the warning says which key, what it said, what is in force, and
    the rule that made the difference — never just "ignored".
    """

    leaf: str
    configured: int
    in_force: int
    rule: str

    def describe(self) -> str:
        """One clause for the read path's single warning line."""

        return (
            f"{self.leaf}: {self.configured} configured, " f"{self.in_force} in force ({self.rule})"
        )


@dataclass(frozen=True)
class ScreeningBudget:
    """The four rolling-hour screening ceilings, as configured or as shipped.

    ``account`` bounds every unit of screening spend. ``ambient`` bounds agent
    chatter in a shared room (tier 0). ``unaddressed`` bounds everything that is
    not a PERSON addressing this agent (tiers 0-2), so the units above it are
    reachable only by the owner's — or another person's — DM or mention.
    ``sender`` bounds ONE verified sender inside the account's window.

    THE INVARIANT IS A NESTING, and the two boundaries answer a contradiction in
    OPPOSITE directions — deliberately, because their duties are opposite:

    * the WRITE path REFUSES a write that would create one (``settings_io
      .validate`` / ``_SCREENING_BUDGET_RULES``), naming the rule while the user
      is still looking at the command that caused it — so "lower the total" is
      answered with which sibling has to come down first, in the same breath;
    * the READ path (a hand-edited YAML, an older config, a file written before
      a rule existed) CLAMPS DOWN, never up: :meth:`clamped` holds each tier to
      the ceiling it is carved out of and adopts the rest of the set exactly as
      written.

    WHY THE READ CLAMPS RATHER THAN DISCARDING THE SET (review round 1, S-2 /
    Q-1). The first version dropped all four to the shipped defaults on any
    contradiction, so the single most likely tightening edit —
    ``budget_per_hour: 10`` written above its siblings — made the receiver spend
    10x what the page showed. Clamping down spends LESS than the file states at
    worst and never more, and :meth:`clamped` returns what it held so the one
    warning can name each key and the value actually in force.

    PER-FIELD RULE (the module's ordinary convention for a present-but-unusable
    value): a value that is missing, not an ``int``, or below 1 reads the SHIPPED
    default for THAT field only. ``bool`` is excluded by name: it is an ``int``
    to Python, so ``true`` would otherwise read as the tightest possible ceiling
    (1) from a value nobody meant as a number.

    THE READ'S UPPER BOUND is :data:`SCREENING_BUDGET_MAX_READ`, the registry's
    own maximum for these rows, and it is the only clamp that is not about the
    nesting: the settings page can show and edit 1..100_000, so a hand edit
    above it would otherwise put the receiver somewhere no surface in the product
    can reach to lower again.
    """

    account: int = DEFAULT_TEAM_BRIDGE_SCREENING_BUDGET_PER_HOUR
    ambient: int = DEFAULT_TEAM_BRIDGE_SCREENING_AMBIENT_BUDGET_PER_HOUR
    unaddressed: int = DEFAULT_TEAM_BRIDGE_SCREENING_UNADDRESSED_BUDGET_PER_HOUR
    sender: int = DEFAULT_TEAM_BRIDGE_SCREENING_SENDER_BUDGET_PER_HOUR

    def __post_init__(self) -> None:
        # The per-field rule applies to EVERY construction path, not only
        # ``from_values``: a caller that builds one from arithmetic must not be
        # able to hand the store a non-int. The CROSS-field rules are NOT
        # enforced here: the WRITE boundary refuses on them (``settings_io
        # .validate``) and the READ boundary holds them down (``clamped``), so
        # neither has to accept a constructor that rewrites a number nobody
        # asked it to rewrite.
        object.__setattr__(
            self,
            "account",
            _screening_ceiling(self.account, DEFAULT_TEAM_BRIDGE_SCREENING_BUDGET_PER_HOUR),
        )
        object.__setattr__(
            self,
            "ambient",
            _screening_ceiling(self.ambient, DEFAULT_TEAM_BRIDGE_SCREENING_AMBIENT_BUDGET_PER_HOUR),
        )
        object.__setattr__(
            self,
            "unaddressed",
            _screening_ceiling(
                self.unaddressed, DEFAULT_TEAM_BRIDGE_SCREENING_UNADDRESSED_BUDGET_PER_HOUR
            ),
        )
        object.__setattr__(
            self,
            "sender",
            _screening_ceiling(self.sender, DEFAULT_TEAM_BRIDGE_SCREENING_SENDER_BUDGET_PER_HOUR),
        )

    def clamped(self) -> "tuple[ScreeningBudget, tuple[ScreeningClamp, ...]]":
        """This set held to its nesting — DOWNWARD only — and what was held.

        In containment order (account → unaddressed → ambient, with sender a
        share of the account), and each step compares against the value its
        parent has ALREADY been held to, so a chain of over-large hand edits
        lands on the ceiling the file can actually mean rather than on an
        intermediate its own parent would still refuse.
        """

        held: list[ScreeningClamp] = []
        account = self.account
        if account > SCREENING_BUDGET_MAX_READ:
            held.append(
                ScreeningClamp(
                    SCREENING_BUDGET_LEAF,
                    account,
                    SCREENING_BUDGET_MAX_READ,
                    f"account <= {SCREENING_BUDGET_MAX_READ}",
                )
            )
            account = SCREENING_BUDGET_MAX_READ
        unaddressed = self.unaddressed
        if unaddressed > account:
            held.append(
                ScreeningClamp(
                    SCREENING_UNADDRESSED_BUDGET_LEAF,
                    unaddressed,
                    account,
                    "unaddressed <= account",
                )
            )
            unaddressed = account
        ambient = self.ambient
        if ambient > unaddressed:
            held.append(
                ScreeningClamp(
                    SCREENING_AMBIENT_BUDGET_LEAF,
                    ambient,
                    unaddressed,
                    "ambient <= unaddressed",
                )
            )
            ambient = unaddressed
        sender = self.sender
        if sender > account:
            held.append(
                ScreeningClamp(SCREENING_SENDER_BUDGET_LEAF, sender, account, "sender <= account")
            )
            sender = account
        if not held:
            return self, ()
        return (
            ScreeningBudget(
                account=account, ambient=ambient, unaddressed=unaddressed, sender=sender
            ),
            tuple(held),
        )

    @classmethod
    def read_from_document(
        cls, values: Any
    ) -> "tuple[ScreeningBudget, tuple[ScreeningClamp, ...]]":
        """The ceilings IN FORCE for a whole ``config.yml`` ``values`` mapping.

        The read path's one entry point: it locates :data:`SCREENING_SUBTREE`
        itself (no reader re-spells where screening lives), applies the
        per-field rule and then :meth:`clamped`, so a caller cannot adopt a
        configured set the receiver would refuse to run.
        """

        node = _walk_subtree(values, SCREENING_SUBTREE)
        return cls.from_values({} if node is _MISSING else node).clamped()

    @classmethod
    def from_values(cls, values: Any) -> "ScreeningBudget":
        """The ``team_bridge.screening`` subtree, per-field defaulted.

        Takes the SUBTREE — what ``SCREENING_SUBTREE`` names — not the whole
        document, so a caller cannot pass a shape this reader never sees and
        silently get the shipped ceilings. Anything that is not a mapping (a
        hand-edited ``screening: off``) reads as the shipped ones. The
        CROSS-FIELD action belongs to :meth:`clamped`; a reader that wants the
        set it may run asks :meth:`read_from_document`.
        """

        node = values if isinstance(values, Mapping) else {}
        return cls(
            account=node.get(SCREENING_BUDGET_LEAF, _MISSING),
            ambient=node.get(SCREENING_AMBIENT_BUDGET_LEAF, _MISSING),
            unaddressed=node.get(SCREENING_UNADDRESSED_BUDGET_LEAF, _MISSING),
            sender=node.get(SCREENING_SENDER_BUDGET_LEAF, _MISSING),
        )


@dataclass(frozen=True)
class TeamBridgeSettings:
    """The local keys, defaulted and normalised.

    Five: ``enabled`` and ``session`` are the registry's editable rows, and
    ``onboarding`` (the one-time dialog's shown-flag, §5.3) is internal state —
    deliberately NOT a registry row: it is UI bookkeeping, not a knob, and a
    public row would dress session-format state as configuration.
    ``default_enrollment`` is the same kind of internal state: the receipt that
    an earlier opt-out default consented once, not a user preference; under the
    consent hold nothing writes it and it drives no consent. SCREENING IS
    NOT HERE AT ALL: it is the admission gate (§8, §13.9), an invariant the
    implementation enforces — not a setting any write can turn off.
    """

    enabled: bool = DEFAULT_TEAM_BRIDGE_ENABLED
    onboarding: str = DEFAULT_TEAM_BRIDGE_ONBOARDING
    session: str = DEFAULT_TEAM_BRIDGE_SESSION
    default_enrollment: Mapping[str, str] = field(default_factory=dict)
    #: Minutes a held conversation message ages before the owner is told; 0 = off.
    hold_alert_after_minutes: int = DEFAULT_HOLD_ALERT_AFTER_MINUTES
    #: B4b owner release (``hold_release.py``); OFF unless a real ``true``.
    hold_release: bool = DEFAULT_HOLD_RELEASE
    #: Whether this person's composers report "is typing" (sender-side only).
    share_typing: bool = DEFAULT_TEAM_BRIDGE_SHARE_TYPING
    #: Whether ``team_bridge.enabled`` was PRESENT in the document (any value,
    #: including a hand-edited ``null``, which reads OFF). It is what separates
    #: "the user said so" from "nobody said anything", the one distinction the
    #: default-on gate needs: an explicit value always wins over a default.
    enabled_is_set: bool = False

    def enabled_given(self, default_on_in_force: bool) -> bool:
        """The composed local switch: explicit value, else the served default.

        An explicit ``true`` or ``false`` (or an unreadable present value, which
        reads OFF) is returned as is. Only an ABSENT key consults the hub: it
        reads ON while ``default_on_in_force`` (the hub serves a member of
        ``consent_versions.DEFAULT_ON_CONSENT_VERSIONS`` and lists it as
        accepted) and OFF otherwise, so a
        v1-only hub sees exactly what :attr:`enabled` says today. ON here means
        "eligible to be enrolled", never "authorised": every send, receive and
        wake path still needs the committed owner and a current receipt.
        """

        if self.enabled_is_set:
            return self.enabled
        return bool(default_on_in_force)

    @staticmethod
    def from_config(cm: "ConfigManager") -> "TeamBridgeSettings":
        return TeamBridgeSettings._derive(cm.get_nested_value)

    @staticmethod
    def from_values(values: Mapping[str, Any]) -> "TeamBridgeSettings":
        """Same derivation over a bare ``values`` mapping (a config document)."""

        def get(path: tuple[str, ...], default: Any) -> Any:
            node: Any = values
            for step in path:
                if not isinstance(node, Mapping) or step not in node:
                    return default
                node = node[step]
            return node

        return TeamBridgeSettings._derive(get)

    @staticmethod
    def _derive(get: Callable[..., Any]) -> "TeamBridgeSettings":
        """The one place that says how the keys are typed and defaulted."""

        raw_enabled = get(ENABLED_PATH, _MISSING)
        return TeamBridgeSettings(
            enabled=_enabled(raw_enabled),
            enabled_is_set=raw_enabled is not _MISSING,
            onboarding=_onboarding(get(ONBOARDING_PATH, DEFAULT_TEAM_BRIDGE_ONBOARDING)),
            session=_session(get(SESSION_PATH, DEFAULT_TEAM_BRIDGE_SESSION)),
            default_enrollment=_default_enrollment(
                get(DEFAULT_ENROLLMENT_PATH, DEFAULT_TEAM_BRIDGE_ENROLLMENT)
            ),
            hold_alert_after_minutes=_hold_alert_after_minutes(
                get(HOLD_ALERT_AFTER_PATH, DEFAULT_HOLD_ALERT_AFTER_MINUTES)
            ),
            # ``is True``: a release path opens on an explicit boolean only.
            hold_release=get(HOLD_RELEASE_PATH, DEFAULT_HOLD_RELEASE) is True,
            share_typing=_share_typing(get(SHARE_TYPING_PATH, _MISSING)),
        )


def record_default_enrollment(
    config_dir: "Path", account_id: str, value: str = DEFAULT_ENROLLMENT_DONE
) -> None:
    """Record a completed default-enrollment consent write for ``account_id``.

    Retained through the consent hold: no production path calls this while the
    status read is forbidden from enrolling, but the stored marker keeps its
    reader and a future consent version may reuse the shape. It must never be
    what authorizes a consent write.

    Written through ``settings_io._store`` because this is INTERNAL state with
    no registry row (the same reason ``onboarding`` is hand-written): a public
    row would dress a one-time receipt as a knob. The import stays FUNCTION-LOCAL
    because ``settings_io`` is loaded on every CLI start and this module is on
    that path (see the module docstring); ``_store`` is the one merge-preserving
    nested writer, so a concurrent hand edit to a sibling ``team_bridge`` key is
    not clobbered by this write.

    Read-merge-write runs under the config write lock (not just the leaf write):
    two members enrolling on the same machine must both land in the mapping, and
    the second must not overwrite the first's entry. Best-effort by contract:
    the caller owns the enrollment itself and must not fail a status read
    because a bookkeeping write could not land.
    """

    if not isinstance(account_id, str) or not account_id.strip():
        raise ValueError("account_id must be a non-empty string")
    if value != DEFAULT_ENROLLMENT_DONE and parse_pending_default_enrollment(value) is None:
        raise ValueError("value must be 'done' or a pending:<endpoint_id>:<head>:<version> marker")

    from local_operator import settings_io
    from local_operator.config import CONFIG_FILE_NAME, ConfigManager
    from local_operator.config_writer import config_write_lock

    with config_write_lock(Path(config_dir)):
        # The SAME pre-write guard every settings write uses: an unreadable
        # document must abort before ConfigManager's recovery rename moves the
        # user's file aside. A missing file is a first run and is allowed.
        settings_io.require_readable_config_path(Path(config_dir) / CONFIG_FILE_NAME)
        manager = ConfigManager(config_dir)
        current = dict(TeamBridgeSettings.from_config(manager).default_enrollment)
        # ``done`` is terminal: a pending marker landing after a Turn off (which
        # writes ``done``) must never reopen the retry it closed.
        if current.get(account_id) in (DEFAULT_ENROLLMENT_DONE, value):
            return
        current[account_id] = value
        settings_io._store(manager, DEFAULT_ENROLLMENT_PATH, current)


def _store_internal(config_dir: "Path", path: tuple[str, ...], value: Any) -> None:
    """Write one internal ``team_bridge.*`` leaf under the config write lock.

    The same guarded, merge-preserving path :func:`record_default_enrollment`
    uses (see its docstring for the lock, the pre-write readability guard and
    the function-local imports); kept separate because these writers store a
    scalar, not a read-merge-write mapping.
    """

    from local_operator import settings_io
    from local_operator.config import CONFIG_FILE_NAME, ConfigManager
    from local_operator.config_writer import config_write_lock

    with config_write_lock(Path(config_dir)):
        settings_io.require_readable_config_path(Path(config_dir) / CONFIG_FILE_NAME)
        settings_io._store(ConfigManager(config_dir), path, value)


def record_local_enabled(config_dir: "Path", value: bool) -> None:
    """Persist an EXPLICIT ``team_bridge.enabled`` from the runtime.

    The one-click turn-off (``False``, the sticky local opt-out every fence and
    the enrolment gate read) writes it after the server answered. The automatic
    enrolment does NOT use this for ``True``: it must never overwrite an
    explicit value, see :func:`record_local_enabled_if_unset`. Best-effort by
    contract at the call sites.
    """

    if not isinstance(value, bool):
        raise ValueError("value must be a bool")
    _store_internal(config_dir, ENABLED_PATH, value)


def record_local_enabled_if_unset(config_dir: "Path") -> bool:
    """Write ``team_bridge.enabled = true`` ONLY while the key is still absent.

    WHY NOT :func:`record_local_enabled`. The automatic enrolment decides to
    write after an awaited hub round trip; an explicit value (``false`` from a
    settings edit or the one-click Turn off, or an explicit ``true``) may have
    landed in that window, and "explicit false always wins" means the default
    must never overwrite it. The check and the write share ONE config write lock
    (reentrant with the leaf write), so no writer can slip in between them.
    Returns whether it wrote. Best-effort at the call site, like its sibling.
    """

    from local_operator import settings_io
    from local_operator.config import CONFIG_FILE_NAME, ConfigManager
    from local_operator.config_writer import config_write_lock

    with config_write_lock(Path(config_dir)):
        settings_io.require_readable_config_path(Path(config_dir) / CONFIG_FILE_NAME)
        manager = ConfigManager(config_dir)
        if TeamBridgeSettings.from_config(manager).enabled_is_set:
            return False
        settings_io._store(manager, ENABLED_PATH, True)
        return True


def record_local_session_if_unset(config_dir: "Path") -> bool:
    """Designate the account's chief-of-staff session ONLY while none is chosen.

    WHY (operator incident, 2026-10-28). A teammate's bridge delivered to a task
    session ("Hyundai POC") instead of her chief of staff: whatever session was
    on screen when Team Bridge was switched on became the wake destination, and
    nothing on the enrolment paths ever looked up the chief of staff. The
    designation is the wake consent (§13.6), so it must name a session the
    person would expect to be woken — and the one durable answer to "who is my
    chief of staff on this machine" is Aida's own record
    (:func:`local_operator.aida.state.session_id_of`), which survives renames.

    THE RULE. Write ``team_bridge.session`` = that session id when the key is
    ABSENT, under the config write lock, and never over an explicit choice: a
    person who picked a session in Settings keeps it. A person who CLEARED it is
    also an explicit choice, and that is why the key is stored empty rather than
    deleted (``settings_io``'s ``empty_unsets=False``): "absent" can only then
    mean "never touched", so a withdrawn wake consent is never re-granted by a
    later enable. A key deleted by an older build (before that change) is
    indistinguishable from one never set; the next enable defaults it once, which
    is the same default a fresh machine gets. It writes the SETTING, not the
    store, because the status read's ``_sync_designation`` mirrors the setting
    into the native record and
    would clear a store-only value on the next read. No Aida session on this
    machine → nothing is written (badge only, as before). Returns whether it
    wrote; best-effort at the call sites, like its siblings.
    """

    from local_operator import settings_io
    from local_operator.aida.state import session_id_of
    from local_operator.config import CONFIG_FILE_NAME, ConfigManager
    from local_operator.config_writer import config_write_lock

    chief = session_id_of(config_dir)
    if not chief:
        return False
    with config_write_lock(Path(config_dir)):
        settings_io.require_readable_config_path(Path(config_dir) / CONFIG_FILE_NAME)
        manager = ConfigManager(config_dir)
        # Presence matters: an explicit empty/null is badge-only consent, not
        # an invitation to choose a wake destination on the person's behalf.
        if manager.get_nested_value(SESSION_PATH, _MISSING) is not _MISSING:
            return False
        settings_io._store(manager, SESSION_PATH, chief)
        return True


def _share_typing(raw: Any) -> bool:
    """Absent → the shipped default (ON); present → ON only for a real ``true``."""

    if raw is _MISSING:
        return DEFAULT_TEAM_BRIDGE_SHARE_TYPING
    return raw is True


def record_share_typing(config_dir: "Path", value: bool) -> None:
    """Persist the person's typing-indicator sharing choice (local only).

    Always writes an explicit boolean, including ``true``: once the person has
    touched the toggle their answer is recorded rather than left to track a
    future default change.
    """

    _store_internal(config_dir, SHARE_TYPING_PATH, bool(value))


def record_onboarding_seen(config_dir: "Path") -> None:
    """Mark the one-time notice dismissed on this machine (never un-set here)."""

    _store_internal(config_dir, ONBOARDING_PATH, ONBOARDING_SEEN)
