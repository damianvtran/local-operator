"""Why a connector failure needs a person, as types rather than prose.

`tunnels/api.py` deliberately refuses to echo anything a provider said — a
response body can carry a bearer, a client secret or a connector token — so
every failure it raises carries a fixed literal sentence. That redaction rule
means the *message* cannot be what a caller classifies on: it says "the login
could not be refreshed" for a network fault and for a revoked grant alike,
which is exactly the misdirection that sent a lost network to /login.

The exception class is the signal instead. Each type here answers one question
the supervisor has to get right: can a retry help at all, or does this state
need an act that only the operator can perform? A type per answer keeps the
classification free of substring matching against wording that is free to
change, and keeps it in ONE place (`tunnels/service.classify_failure`) rather
than in each module that happens to raise it.
"""

from __future__ import annotations


class LoginRequired(ValueError):
    """The tunnel's Radient login is unusable; only a re-login can fix it.

    Raised where the credential store has already made that judgement — it is
    the wrapper `tunnels/api.py` puts around
    :class:`~local_operator.providers.auth_store.CredentialInvalidError`, and
    the type `is_terminal_grant_response` earns through it.

    A subclass of ``ValueError`` on purpose: every existing handler on this
    path (`tunnels/cli.py`, `tunnels/service.main`) catches ``ValueError``, and
    none of them should have to learn a new base class to keep working.
    """


class LocalPrerequisite(ValueError):
    """A local prerequisite is missing, so no retry can change the outcome.

    Missing or too-old cloudflared, no tunnel configuration on this device, an
    uninstalled mobile relay. The remedy is a local command
    (``lop tunnel install`` / ``lop tunnel create`` / ``lop mobile install``),
    and each of those re-arms a parked connector, so parking is recoverable
    without the operator doing anything beyond the fix they already had to make.
    """


class ReenrolmentRequired(ValueError):
    """The cloud's own answer invalidates this device's enrolment.

    A console-side gateway-port or harness-port change, or a harness the
    console added: the connector must not accept the new shape (see
    ``service.enforce_harness_ports``) and the same command that renews the
    local record — ``lop tunnel connect`` — is what clears the park.
    """
