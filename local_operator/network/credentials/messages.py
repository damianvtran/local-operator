"""The one home for what a person reads when a borrow is refused (design §4/§2.5).

Every sentence the credential broker can put in front of a human is spelled here
and NOWHERE else. The reason is the one ``relay._owning_document`` gives for its
own table: a refusal that is written at three call sites drifts into three
refusals, and the drift is invisible until an operator is told to run a command
that does not exist.

WHAT EVERY SENTENCE MUST DO, because these are read mid-task:

* name the DEVICE that owns the credential, by the name the operator gave it;
* say what to do NEXT, and say the local remedy only where there is one — "run
  ``lop login openai`` here to use your own account" is real advice, and
  "contact support" would not be;
* never name a token, key, prefix or hash. Not one of these functions takes a
  credential VALUE, which is the structural half of that rule: there is nothing
  to accidentally interpolate.

The codes themselves are closed (``types.BROKER_ERROR_CODES``); an unknown code
falls through to :func:`_generic`, which still names the owner — a refusal that
tells the operator nothing is the failure mode this module exists to prevent.
"""

from __future__ import annotations

from local_operator.network.credentials.types import BrokerError


def _owner_name(error: BrokerError, fallback: str) -> str:
    return error.owner_device_name or error.owner_device or fallback


def render_last_seen(seconds: float | None) -> str:
    """``4 min ago`` / ``just now`` / ``never`` — for the owner-offline sentence.

    Round numbers and one unit, deliberately: the operator needs to judge "is that
    machine asleep or did I shut it down", and seconds-of-precision would suggest
    this is a measurement when it is an observation with a heartbeat behind it.
    """
    if seconds is None:
        return "never"
    if seconds < 90:
        return "just now"
    minutes = seconds / 60.0
    if minutes < 90:
        return f"{int(round(minutes))} min ago"
    hours = minutes / 60.0
    if hours < 36:
        return f"{int(round(hours))} h ago"
    return f"{int(round(hours / 24.0))} d ago"


def render_broker_error(
    error: BrokerError,
    *,
    key: str = "",
    owner_name: str = "the owner device",
    last_seen_s: float | None = None,
    provider: str = "",
) -> str:
    """The sentence for ``error``, with the remedy its code implies.

    ``key`` is the placement key the borrow was for; ``provider`` is the provider
    the operator would type at ``lop login``. They differ for MCP, where the key
    is ``mcp:<url>`` and there is no provider verb — which is why the MCP arms say
    ``/mcp login`` and not ``/login``.
    """
    owner = _owner_name(error, owner_name)
    label = provider or key or "that credential"
    is_mcp = key.startswith("mcp:")
    login = f"/mcp login {key[4:]}" if is_mcp else f"lop login {label}"

    if error.code == "owner_offline":
        return (
            f"No credential for '{label}' is reachable: {owner} owns it and was last seen "
            f"{render_last_seen(last_seen_s)}. Reconnect that device, or run '{login}' here "
            "to use your own account."
        )
    if error.code == "not_a_holder":
        return (
            f"{owner} does not share '{label}' with this device. Ask the operator on "
            f"{owner} to share it (on that device: 'lop network credential share {label} "
            "--with <this device>'), or run '"
            f"{login}' here to use your own account."
        )
    if error.code == "not_owner":
        return (
            f"{owner} does not hold '{label}': the device that signed in to it is another "
            "one. Nothing was changed; ask the device that owns it to share it, or run "
            f"'{login}' here to use your own account."
        )
    if error.code == "revoked":
        return (
            f"{owner} no longer shares '{label}' with this device. Nothing was changed here; "
            f"if that was not intended, share it again on {owner}."
        )
    if error.code == "grant_invalid":
        return (
            f"the credential for '{label}' that {owner} lends out was refused by the "
            f"provider and cannot be revived by retrying. Re-run '{login}' on {owner}. "
            "Nothing is disabled on either device."
        )
    if error.code == "device_bound":
        return (
            f"{label} logins are bound to the device that made them, so {owner} cannot lend "
            f"this one; run '{login}' on the device that needs it."
        )
    if error.code == "quota_blocked":
        return (
            f"'{label}' on {owner} is rate-limited for now, so nothing was borrowed. The "
            "usual failover to another of your own logins is already running."
        )
    if error.code == "interactive_required":
        # "can run", not "has been asked to": nothing asks the owner (review round 2, M3 —
        # the design's §4.7 ``credential_repair`` op is not built, which is what the
        # amended row records). A sentence promising a request nobody sent is the same
        # class of drift the amendment exists to remove.
        return (
            f"'{label}' needs an interactive sign-in on {owner} before it can be lent out; "
            f"the operator there can run '{login}'."
        )
    if error.code == "epoch_stale":
        return (
            f"{owner} and this device disagree about the network's epoch, so nothing was "
            "borrowed. Reconnect and try again."
        )
    if error.code == "refresh_failed":
        return (
            f"{owner} could not refresh '{label}' just now (a transient provider or network "
            "failure). This is a retryable outage, not a broken login."
        )
    if error.code == "rate_limited":
        return (
            f"{owner} is handling too many credential requests at the moment and declined this "
            "one rather than queueing it. Try again shortly."
        )
    if error.code == "not_authorised":
        return (
            f"{owner} refused this request: this device is not allowed to ask for '{label}'. "
            "Nothing was changed on either device."
        )
    if error.code == "identity_mismatch":
        return (
            f"{owner} refused this request because it named a different sending device than "
            "the one it arrived from. Nothing was lent or changed on either device; if this "
            "repeats, update both devices to the same build."
        )
    if error.code == "unsupported":
        return (
            f"{owner} runs a build that cannot lend credentials, so '{label}' cannot be "
            "borrowed from it. This behaves exactly as it did before the mesh: the request "
            "runs with the logins this device has."
        )
    if error.code == "no_local_credential":
        return (
            f"{owner} holds no usable credential for '{label}' either, so there is nothing to "
            f"borrow. Run '{login}' on {owner} to sign in there."
        )
    if error.code == "not_implemented":
        return (
            f"this build cannot lend credentials yet ({owner} answered that the broker is not "
            "implemented), so '{label}' was not borrowed. Nothing was changed."
        )
    return _generic(error, label=label, owner=owner, login=login)


def _generic(error: BrokerError, *, label: str, owner: str, login: str) -> str:
    """The last resort: still names the owner, the credential and the remedy.

    Reached for a code this build does not know, which means a NEWER owner
    refused for a reason we cannot classify. Saying so is more useful than
    rendering the owner's internal code at the operator.
    """
    detail = f" ({error.message})" if error.message else ""
    return (
        f"{owner} declined to lend '{label}'{detail}. Nothing was changed on either device; "
        f"run '{login}' here to use your own account."
    )


def render_borrowed_signin(owner: str, name: str) -> str:
    """The borrower's line when an MCP server refuses a BROKERED sign-in.

    WHY THIS EXISTS (review round 1, R1 — audit round 2's F5). ``_auth_required_text``
    answered every 401-without-a-local-grant with ``/mcp login <name> to authorize``,
    and on a BORROWING device that command is wrong twice over: a headless borrower
    cannot open a browser at all, and a borrower with one creates a LOCAL grant that
    wins over the borrow (``manager._brokered_mcp_auth``'s local-wins rule) — a silent
    account switch. The sign-in this failure is about lives on ``owner``, so the line
    names the owning device and puts the command THERE.

    Short on purpose: this string is the tail of the toast's ``failed: <name> — …``,
    whose budget has already paid for the label — measured at the widest card as 56
    cells (``TOAST_MAX_WIDTH 60 - TOAST_PADDING_CELLS 2`` minus the detail row's own
    2), of which the label spends 17 for ``linear`` (review round 2, M1: the first
    version of this sentence was 55 cells and composed into a 72-cell row, so the
    command the operator was told to run was cut to ``/mcp l…``).

    The COMPOSED row is the contract, not this string: 38 cells here and 55 composed
    for the canonical pair (19-cell device name ``damians-MacBook-Pro``, 6-cell
    server ``linear``), which fits. Longer names do not, and then
    ``toast._fit_failure_line`` keeps the OWNER and the command's HEAD and sheds the
    server name the command repeats from the row's own label —
    ``failed: launchdarkly — damians-MacBook-Pro: /mcp login …``, 56 cells against the
    widest card's budget (design round 1, D1: the owner is WHERE the sign-in is, so a
    row that sheds it prints the local instruction this family exists to replace,
    while the elided argument is the one part the row already carries in its label).
    Only below THAT does the command outrank the owner, and then it is the command
    base printed (design round 1, D2). The owner still leads whenever both fit,
    because the command is only meaningful on that device: a reader who cannot see
    WHERE it runs may run it here.
    """
    return f"{owner}: /mcp login {name}"


def render_success(provider: str, owner_name: str, *, cached: bool = False) -> str:
    """What `lop network credential ... --json`-less output says about a live borrow."""
    verb = "using the borrowed" if cached else "borrowed"
    return (
        f"{verb} '{provider}' from {owner_name} "
        "(access only; no refresh token leaves that device)"
    )


#: The WIRE DIAGNOSTIC an owner sends with an ``interactive_required`` refusal.
#:
#: WHAT IT IS NOT: the sentence the borrower's operator reads. Since review round 1
#: (audit round 2, F1) the requester renders EVERY code from this module, so an
#: owner's own words reach a person only through :func:`_generic`'s parenthetical,
#: where they are the detail of a code this build cannot classify. This string's
#: reader is therefore a human reading a wire capture or the owner's logs, not a
#: session's operator.
#:
#: The "owner-side toast" a first draft of this docstring described does not exist:
#: the design's §4.7 row asks for a ``credential_repair`` op that raises a durable
#: notice on the owning device, and NO SUCH OP IS BUILT (grep ``credential_repair``:
#: nothing outside that design row). What the owner has today is the
#: ``credential.report`` audit record written beside this refusal, and what the
#: borrower has is the ``interactive_required`` sentence in
#: :func:`render_broker_error` — which names the owner and the command to run there,
#: and is the reason the composed MCP failure line can be truthful without a repair
#: op.
#:
#: ``key`` is the placement key (``mcp:<url>`` or a provider name); the quoting is
#: deliberate, because the operator pastes the command.
def render_repair_notice(peer_name: str, key: str) -> str:
    is_mcp = key.startswith("mcp:")
    verb = f"/mcp login {key[4:]}" if is_mcp else f"/login {key}"
    return f"{peer_name} needs '{verb}' here — its borrowed credential cannot be refreshed"
