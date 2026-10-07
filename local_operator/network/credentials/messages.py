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

from typing import TYPE_CHECKING

from local_operator.network.credentials.github import GITHUB_KEY, no_source_arms
from local_operator.network.credentials.types import BrokerError

if TYPE_CHECKING:  # pragma: no cover - typing only
    from local_operator.session.credential_binding import CredentialBinding


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


def render_retry_after(ms: float | int | None) -> str:
    """``45 s`` / ``2 min`` — the forward twin of :func:`render_last_seen`.

    Round numbers, one unit, for the same reason as there: the operator is judging
    "retry now or move on", and a second-by-second figure would read as a
    measurement rather than as the hint the owner actually sent. The value it
    renders was clamped at the wire boundary when it came from an owner
    (``MAX_PEER_RETRY_AFTER_MS``) and is counted down by
    ``MeshCredentialClient.cached_refusal`` only when one was stated, so the
    sentence never promises more than the cache will honour. ``""`` means "nothing
    to say" — no owner value was sent, or the boundary dropped one.
    """
    if not ms or ms <= 0:
        return ""
    seconds = float(ms) / 1000.0
    if seconds < 60:
        # Five-second steps: the coarsest step that still tells a 45 s wait apart
        # from a 5 s one, and never a ``0 s``.
        return f"{max(5, int(seconds / 5.0 + 0.5) * 5)} s"
    return f"{max(1, int(seconds / 60.0 + 0.5))} min"


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
    is_github = provider == GITHUB_KEY or key == GITHUB_KEY

    if is_github:
        # THE GITHUB ARMS COME FIRST, and each says only what is true for this
        # credential. It has no local-login remedy (there is no 'lop login github'
        # in this build), and after the source ladder (§3.2) its "the owner has
        # nothing" state means NO arm is configured — so the remedy sentence
        # composes around ``no_source_arms()``, the ONE enumeration the device's
        # own refusal also uses (review round 1, n1: the list cannot drift).
        if error.code == "no_local_credential":
            return (
                f"{owner} holds no GitHub credential: push and PR-write through the mesh "
                f"are unavailable until one is configured there ({no_source_arms()}). "
                "Public clones and non-GitHub work are unaffected."
            )
        if error.code == "not_a_holder":
            return (
                f"{owner} does not share its GitHub credential with this device. Ask "
                f"the operator on {owner} to share it (on that device: 'lop network "
                "credential share github --with <this device>'). Public clones and "
                "non-GitHub work are unaffected."
            )
        if error.code == "owner_offline":
            return (
                f"No GitHub credential is reachable: {owner} owns it and was last "
                f"seen {render_last_seen(last_seen_s)}. Reconnect that device; public "
                "clones and non-GitHub work are unaffected."
            )

    if error.code == "device_scope_required":
        return (
            f"{owner} refused to lend '{label}' to a session: this credential is "
            "bound to the DEVICE, not to a session id (any process on the borrowing "
            "device may use it while the share stands). Nothing was lent; re-share "
            f"it with 'lop network credential share {label} --with <device> --scope "
            "device'."
        )
    if error.code == "github_app_unusable":
        return (
            f"{owner} could not use its GitHub App credential, so nothing was minted "
            f"(re-store the key there: 'lop secret update GITHUB_APP' on {owner} — the "
            "network guide has the App section). Nothing was lent."
        )
    if error.code == "github_repositories_unset":
        return (
            f"{owner} has no repositories designated for GitHub brokering, so nothing "
            f"was minted: set network.credentials.github.repositories on {owner} (the "
            "network guide's checklist). Nothing was lent."
        )
    if error.code == "github_repo_refused":
        detail = f" ({error.message})" if error.message else ""
        return (
            f"GitHub refused the mint for a repository {owner} designated{detail}. "
            f"Check that the App installation on {owner} covers the designated "
            "repositories; nothing was lent."
        )
    if error.code == "github_token_unusable":
        return (
            f"{owner} could not use the GITHUB_TOKEN-class credential it stores, so "
            f"nothing was lent: re-store the token on {owner} (the network guide has "
            "the ladder), or set up the stronger GitHub App there."
        )
    if error.code == "github_gh_unusable":
        return (
            f"{owner} could not use the login its gh CLI stores, so nothing was lent: "
            f"the owner needs to sign in again with the gh CLI (the network guide has "
            "the ladder). Push and PR-write through the mesh are unavailable until "
            "that source works or another one is configured; public clones and "
            "non-GitHub work are unaffected."
        )
    if error.code == "github_store_unreadable":
        return (
            f"{owner}'s encrypted secret store could not be read, so no GitHub "
            "credential could be resolved there: nothing was lent, and no wider source "
            f"is served while the store is unreadable (repair or restore it on {owner} "
            "— the network guide has the ladder)."
        )

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
        # THE REMAINDER, WHEN THE OWNER MEASURED ONE (design round 1, D1). The
        # producer sends the earliest unblock and the cache keeps counting it down;
        # "for now" read as if nothing were known, and the design's §4.9 row shows a
        # time for exactly this refusal. Absent (an older owner, or a value the
        # boundary dropped) keeps the old wording.
        left = render_retry_after(error.retry_after_ms)
        if left:
            return (
                f"'{label}' on {owner} is rate-limited for another {left}, so nothing "
                "was borrowed. The usual failover to another of your own logins is "
                "already running."
            )
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
    if error.code == "local_only":
        # §4.2's kill switch, said where a would-be borrower meets it: the mark
        # is the operator's, on the owning device, and the remedy is named
        # there rather than here.
        return (
            f"'{label}' is marked local-only on {owner}, so it never crosses to another "
            "device. Nothing was copied; if it should travel, remove the mark on "
            f"{owner} ('lop network credential mark {key} default')."
        )
    if error.code == "copy_stale":
        # THE COPY PATH'S ONE ADDED CODE (design §6.1): the value this device
        # holds is older than the owner's and the provider refused it. The
        # remedy is the owner's own next contact; the ending, if it never
        # heals, is the §2.3 ceiling sentence — a copied secret ends at its
        # source, not at the device.
        return (
            f"the stored copy of '{label}' is older than {owner}'s, and the provider "
            f"refused it. {owner} announces the current value on its next contact — "
            "nothing is blocked meanwhile; if it keeps failing, rotate the secret at "
            "its source."
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


#: The interactive login that clears an ``interactive_required`` repair — the ONE
#: spelling of the command, shared by the wire diagnostic below, the owner-side
#: repair notice (``credentials/repair.py``) and its doctor row's remedy: three
#: surfaces tell a reader to run the same thing, so the thing is spelled once.
def repair_command(key: str) -> str:
    is_mcp = key.startswith("mcp:")
    return f"/mcp login {key[4:]}" if is_mcp else f"/login {key}"


#: The WIRE DIAGNOSTIC an owner sends with an ``interactive_required`` refusal —
#: and the wording the owner-side repair notice reuses (``credentials/repair.py``),
#: so the wire and the owner's own surfaces say one thing.
#:
#: WHAT IT IS NOT: the sentence the borrower's operator reads. Since review round 1
#: (audit round 2, F1) the requester renders EVERY code from this module, so an
#: owner's own words reach a person only through :func:`_generic`'s parenthetical,
#: where they are the detail of a code this build cannot classify. On the wire, this
#: string's reader is a human reading a wire capture or the owner's logs.
#:
#: The "owner-side toast" a first draft of this docstring described does not exist:
#: the design's §4.7 row asked for a ``credential_repair`` op that raises a durable
#: notice on the owning device, and NO SUCH OP IS BUILT. What ships instead is
#: DERIVED from the ``credential.report`` audit record written beside this refusal:
#: ``credentials/repair.py`` turns an open report into a ``credential_repair`` check
#: row that ``lop network doctor`` (relay path and local fallback) and the
#: ``/network`` panel show on the OWNER — which is the sense in which this wording
#: now reaches an operator, recomposed on their own surface rather than sent to
#: them. What the borrower has is the ``interactive_required`` sentence in
#: :func:`render_broker_error` — which names the owner and the command to run there,
#: and is the reason the composed MCP failure line can be truthful without a repair
#: op.
#:
#: ``key`` is the placement key (``mcp:<url>`` or a provider name); the quoting is
#: deliberate, because the operator pastes the command.
def render_repair_notice(peer_name: str, key: str) -> str:
    return (
        f"{peer_name} needs '{repair_command(key)}' here — "
        "its borrowed credential cannot be refreshed"
    )


#: THE ONE SENTENCE THE DESIGN COMMITS TO (§2.3), on every surface that ends a
#: copy: share receipt, guide, and `credential revoke` output. It is not a
#: disclaimer — it states the actual limit of a wipe (reachable copies end; an
#: exfiltrated one does not) and the only effective ending (rotation at the
#: source).
COPY_CEILING_SENTENCE = (
    "This removed the copies it could reach. A copy that already left that device "
    "can only be ended by rotating the secret at its source."
)


def render_copy_revoke_notice(peer_name: str, key: str, *, copied: bool, wiped: bool) -> str:
    """What a revoke says about the COPY half, per state (§4.3, "which happened").

    ``copied`` — the ack ledger says this device confirmed holding a copy;
    ``wiped`` — it has since confirmed deleting it. Both come from the same
    ledger the owner's listing reads, so the receipt and the listing cannot
    disagree about whether a copy is outstanding.
    """
    if not copied:
        return (
            f"no copy of '{key}' was confirmed on {peer_name}, so there is nothing to " "wipe there"
        )
    if wiped:
        return f"the copy of '{key}' on {peer_name} was already wiped"
    return (
        f"a wipe notice for '{key}' is queued for {peer_name}: the copy is deleted on "
        "its next contact"
    )


#: The head every account-change notice carries. The '[session …]' form is this
#: tree's convention for session records that are not conversation (incidents,
#: the MCP warning), and '[session credential]' is the family's own credential
#: head — measured at 20 cells against the first cut's 28, which wrapped the
#: sentence onto a third row at 60 columns (design round 1, D5). The message
#: TYPE and the peek heading ("account change") carry the distinction; nothing
#: consumes the longer words.
_BINDING_NOTICE_HEAD = "[session credential] "


def _device_phrase(name: str, fallback: str) -> str:
    """The owner as a person reads it: the operator's name for the device.

    §2.5's rule — never a device id in a person-facing sentence; the id is
    meaningless to a reader, and when no name is known the sentence says WHAT
    the thing is ("another device") rather than inventing a name for it.
    """
    return name.strip() or fallback


def _login_phrase(label: str, provider: str, owner: str) -> str:
    """The serving login as the row knows it: label first, else provider+owner.

    The ONE noun across the arms is "login" — the word the module's own remedy
    tells the operator to run (`lop login openai`) — and the label (the
    operator-facing account string, when the row carries one) is what answers
    "which of my two logins" when both belong to the same owner.
    """
    if label.strip():
        return f"{label} on {owner}"
    return f"the {provider} login on {owner}"


def render_binding_change_notice(
    binding: CredentialBinding,
    previous: CredentialBinding,
    *,
    self_device: str,
) -> str:
    """The ONE operator-visible notice for an account change (memo D1, M3-M6).

    Written once per appended change row by the recorder's ``on_change`` seam
    (``Session.journal_credential_binding_change``); the row persists it to the
    transcript, and it is deliberately NOT model-visible (see
    ``SESSION_BINDING_NOTICE_MESSAGE_TYPE``).

    Returns ``""`` — "nothing a person must be told", a label-only refinement
    of the same account — and the caller journals nothing; the ROW still
    records the refinement, because the replacement-state rule keeps the
    newest snapshot complete.

    TWO CONSTRAINTS PIN THE SENTENCES (design round 1, D1/D2):

    * **state the fact, not the cause.** The sibling-pick arm fires on ANY
      blocked row — a real quota verdict, a plain 429 rotation, a
      family-scoped block (`model:<family>`, which stops one family and not the
      account), or a remote borrower's report — and the row cannot tell them
      apart, so the sentence may not name a cause only one of them has.
    * **no deictics.** The row is persisted, replayed, and MOVES between
      devices (this slice's E2 cell), so "this turn" and "this device" would
      point at whatever is around them when read back. Every arm is framed as a
      recorded transition and names its devices from the rows themselves.

    ``self_device`` is the recording device's id, which the row cannot carry:
    it is what lets M4 (the session moved onto the recording device's own
    login) be told apart from M6 (the owner moved). The design round owns the
    copy; the STRUCTURE (which change produces which sentence, and that a
    change is never silent) is what the cells pin.
    """
    provider = binding.provider or "credential"
    owner_changed = binding.owner_device != previous.owner_device
    id_changed = binding.credential_id != previous.credential_id
    if not owner_changed and not id_changed:
        # A label refinement, or nothing at all: the same account still serves.
        return ""

    if owner_changed and binding.owner_device == self_device:
        # M4 — the recording device gained its own login and the local-first
        # cascade took the session onto it. Names the account it came FROM and
        # the device it landed on (the row's own name field).
        where = _device_phrase(previous.owner_device_name, "another device")
        was = (
            f"using {previous.identity_label} on {where}"
            if previous.identity_label
            else f"using the {provider} login on {where}"
        )
        landed = _device_phrase(binding.owner_device_name, "the recording device")
        return (
            f"{_BINDING_NOTICE_HEAD}This session was {was}; "
            f"it moved onto {landed}'s own {provider} login."
        )

    if owner_changed and previous.owner_device == self_device:
        # M4's mirror — the local login the session was using is gone (deleted,
        # or no longer resolvable), so the borrow rung answers now.
        was_local = _device_phrase(previous.owner_device_name, "the recording device")
        who = _device_phrase(binding.owner_device_name, "another device")
        now = _login_phrase(binding.identity_label, provider, who)
        return (
            f"{_BINDING_NOTICE_HEAD}This session was running on the {provider} "
            f"login on {was_local}; it is now served by {now}."
        )

    if owner_changed:
        # M6 — an ownership move (the placement ceremony ran elsewhere). Names
        # the NEW owner once, with the old one as context.
        was_owner = _device_phrase(previous.owner_device_name, "another device")
        now_owner = _device_phrase(binding.owner_device_name, "another device")
        now = _login_phrase(binding.identity_label, provider, now_owner)
        return (
            f"{_BINDING_NOTICE_HEAD}This session is now using {now} "
            f"(previously served by {was_owner})."
        )

    if binding.owner_device == self_device:
        # Same device, different row: the local walk re-picked (a re-login, or a
        # sibling the operator added). Rare, and still an account change.
        landed = _device_phrase(binding.owner_device_name, "the recording device")
        return f"{_BINDING_NOTICE_HEAD}This session is now using the {provider} login on {landed}."

    # M3 — the owner-side sibling pick: same owner, a different row. States the
    # FACT of the rotation, never its cause (the row cannot tell the four
    # block writers apart), and names the login it landed on when the row
    # carries a label (design D6). "another", not "its other": the number of
    # logins is unknown, and a third sibling would make "other" a lie
    # (design round 2, N2).
    owner = _device_phrase(previous.owner_device_name, "the owner device")
    landed_on = f" ({binding.identity_label})" if binding.identity_label.strip() else ""
    return (
        f"{_BINDING_NOTICE_HEAD}The {provider} login on {owner} went out of "
        f"rotation, and the next turn ran on another {provider} login{landed_on}."
    )
