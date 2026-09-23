"""Interactive prompt for a provider API key (`lop credential update`).

This was ``CredentialManager.prompt_for_credential``. Its BODY was already
store-only — it saved through :func:`~local_operator.providers.registry.store_provider_key`,
never the plaintext file — so deleting it with ``local_operator.credentials``
would have deleted the only interactive key-entry surface for a name that is not
part of a login flow. It moves here, next to the writer it calls, and its name
loses "credential" so it stops claiming a mechanism the consolidation retired.

The behaviour is preserved byte-for-byte because it is an automation CONTRACT,
not an incidental detail: on a non-tty stdin it reads one line from stdin rather
than calling ``getpass`` (which falls back to echoing with a warning), so
``printf '%s\\n' "$KEY" | local-operator credential update NAME`` keeps working.
"""

import getpass
import sys

from pydantic import SecretStr

from local_operator.cli_style import CYAN, ERROR, SUCCESS, can_encode, paint


def prompt_for_provider_key(env_key: str, reason: str = "not found in configuration") -> SecretStr:
    """Prompt the user for ``env_key`` and save it as a provider-class store row.

    Args:
        env_key: The env-var-spelled name to ask for (``OPENROUTER_API_KEY``).
        reason: Why the prompt is being shown, printed in the box.

    Returns:
        The value the operator entered, wrapped in ``SecretStr``.

    Raises:
        ValueError: If the operator enters an empty key.
        EOFError: If stdin closes before a value is read (piped empty input).
        KeyboardInterrupt: If the operator cancels the prompt.
    """
    # Calculate border length based on key length
    line_length = max(50, len(env_key) + 12)
    # Box drawing is decorative; on a stdout whose encoding cannot represent it
    # (PYTHONIOENCODING=ascii, a legacy Windows code page) drawing it crashed the
    # prompt with UnicodeEncodeError before it could ask for anything. Fall back
    # to ASCII rules so the prompt still works there.
    heavy = can_encode("─╭╮├┤╰╯")
    h, tl, tr, ml, mr, bl, br = (
        ("─", "╭", "╮", "├", "┤", "╰", "╯")
        if heavy
        else (
            "-",
            "+",
            "+",
            "+",
            "+",
            "+",
            "+",
        )
    )
    border = h * line_length

    # Colour is gated on NO_COLOR/tty/TERM by ``paint`` — a raw escape here
    # painted literal ``[1;36m`` into a piped or dumb-terminal transcript.
    def cyan(text: str) -> str:
        return paint(text, CYAN)

    # Print the setup box
    print(cyan(f"{tl}{border}{tr}"))
    setup_padding = " " * (line_length - len(env_key) - 7)
    print(
        cyan(f"│ {env_key} Setup{setup_padding}│")
        if heavy
        else cyan(f"| {env_key} Setup{setup_padding}|")
    )
    print(cyan(f"{ml}{border}{mr}"))
    reason_padding = " " * (line_length - len(env_key) - len(reason) - 3)
    body = f"{env_key} {reason}."
    print(cyan(f"│ {body}{reason_padding}│") if heavy else cyan(f"| {body}{reason_padding}|"))
    print(cyan(f"{bl}{border}{br}"))

    prompt = paint(f"Please enter your {env_key}: ", "1;94")
    if sys.stdin.isatty():
        # Interactive terminal: getpass hides the key so it never lands in
        # scrollback of a session that may be screen-shared.
        credential = getpass.getpass(prompt).strip()
    else:
        # Non-interactive stdin (piped/scripted): getpass on a non-tty falls
        # back to echoing the input with a warning, which is both noisy and
        # useless for automation. Read one line from stdin instead — this is
        # the documented automation contract: `printf '%s\n' "$KEY" |
        # local-operator credential update NAME`. An empty pipe raises
        # EOFError, which the command handler turns into one plain line.
        print(prompt, end="", flush=True)
        line = sys.stdin.readline()
        if line == "":
            raise EOFError(f"{env_key} is required for this step.")
        credential = line.strip()
    if not credential:
        raise ValueError(f"{env_key} is required for this step.")

    try:
        # Lazy, matching the discipline the deleted method used: the registry
        # pulls the provider stack, and this module is reachable from the CLI
        # startup path through ``lop credential update``.
        from local_operator.providers.registry import store_provider_key

        store_provider_key(env_key, credential)
    except Exception as exc:  # noqa: BLE001 - one honest line, not a panel
        from local_operator.ansi import strip_control_sequences

        print(
            paint(
                strip_control_sequences(f"Could not save {env_key}: {exc}"),
                ERROR,
                stream=sys.stderr,
            ),
            file=sys.stderr,
        )
        raise ValueError(f"Could not save {env_key} to the credential store.") from exc

    # ASCII fallback for the check glyph too: a stdout that cannot encode the
    # box drawing cannot encode ✓ either, and crashing on the SUCCESS line
    # after the key is already saved is the worst place to fail (item 14).
    tick = "✓" if can_encode("✓") else "[ok]"
    print(paint(f"\n{tick} Credential successfully saved!", SUCCESS))

    return SecretStr(credential)
